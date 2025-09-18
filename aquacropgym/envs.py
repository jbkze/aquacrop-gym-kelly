from aquacrop.classes import *
from aquacrop.core import *
 
import gym
from gym import spaces

import numpy as np


tunis_maize_config = dict(
    name='tunis_maize',
    weather_df=None, # generated and processed weather dataframe
    crop='Maize', # crop type (str or CropClass)
    planting_date='05/01',
    soil='Sand', # soil type (str or SoilClass)
    dayshift=1, # maximum number of days to simulate at start of season (ramdomly drawn)
    include_rain=False, # maximum number of days to simulate at start of season (ramdomly drawn)
    days_to_irr=7, # number of days (sim steps) to take between irrigation decisions
    max_irr=34, # maximum irrigation depth per event
    max_irr_season=10_000, # maximum irrigation appl for season
    irr_cap_half_DAP=-999, # day after planting to half water supply
    init_wc=InitWCClass(wc_type="Prop", value=["WP"]), # initial water content
    crop_price=180., # $/TONNE
    irrigation_cost = 1.,# $/HA-MM
    fixed_cost = 1728,
    best=np.ones(1000)*-1000, # current best profit for each year
    observation_set='default',
    normalize_obs=True,
    action_set='depth',
    forecast_lead_time=7, # number of days perfect forecast if using observation set x
    CO2conc=363.8,
    
    train_years=[1990, 1991, 1992, 1993, 1994],
    val_years=[1995, 1996, 1997, 1998, 1999],
    test_year=1980,
    manual_year=None
)



class CropEnv(gym.Env):
    """
    AquaCrop-OSPy crop environment in openai-gym style.

    Cropping system consists of a single crop on 1 HA of homogenous soil.
    Each episode consits of 1 season of this system

    Config parameters will specify the type of cropping environment.

    Every `days_to_irr` days, the agent will see an observation of the enviornemnt

    The agent will then make an irrigation decision which is passed to the environment

    This proicess continues until season has finished

    the seasonal profit is calculated and apssed to the agent as the reward
    """
 
    def __init__(self,config):

        
        super(CropEnv, self).__init__()

        self.actions = []
        
        self.weather_df = config["weather_df"]
        self.days_to_irr=config["days_to_irr"]
        self.dayshift = config["dayshift"]
        self.include_rain=config["include_rain"]
        self.max_irr=config["max_irr"]
        self.init_wc = config["init_wc"]
        self.CROP_PRICE=config["crop_price"]
        self.IRRIGATION_COST=config["irrigation_cost"] 
        self.FIXED_COST = config["fixed_cost"]
        self.planting_month = int(config['planting_date'].split('/')[0])
        self.planting_day = int(config['planting_date'].split('/')[1])
        self.max_irr_season=config['max_irr_season']
        self.irr_cap_half_DAP=config['irr_cap_half_DAP']
        self.name=config["name"]
        self.best=config["best"]*1
        self.total_best=config["best"]*1
        self.CO2conc=config["CO2conc"]
        self.observation_set=config["observation_set"]
        self.normalize_obs = config["normalize_obs"]
        self.action_set=config["action_set"]
        self.forecast_lead_time=config["forecast_lead_time"]

        self.train_years=config['train_years']
        self.val_years=config['val_years']
        self.test_year=config['test_year']
        self.manual_year=config['manual_year']

        # crop and soil choice
        crop = config['crop']        
        if isinstance(crop,str):
            self.crop = CropClass(crop,PlantingDate=config['planting_date'])
        else:
            assert isinstance(crop,CropClass), "crop needs to be 'str' or 'CropClass'"
            self.crop=crop

        soil = config['soil']
        if isinstance(soil,str):
            self.soil = SoilClass(soil)
        else:
            assert isinstance(soil,SoilClass), "soil needs to be 'str' or 'SoilClass'"
            self.soil=soil
     
     
        self.tsteps=0

        # observation normalization

        self.mean=0
        self.std=1

        # obsservation and action sets

        if self.observation_set in ['default',]:
            #self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        if self.action_set=='smt4':
            self.action_space = spaces.Box(low=-1., high=1., shape=(4,), dtype=np.float32)

        elif self.action_set=='depth':
            self.action_space = spaces.Box(low=-5., high=self.max_irr+5., shape=(1,), dtype=np.float32)
    
        elif self.action_set=='depth_discreet':
            self.action_depths=[0,2.5,5,7.5,10,12.5,15,17.5,20,22.5,25]
            self.action_space = spaces.Discrete(len(self.action_depths))    

        elif self.action_set=='binary':
            self.action_depths=[0,self.max_irr]
            self.action_space = spaces.Discrete(len(self.action_depths))    

                
    def states(self):
        return dict(type='float', shape=(self.observation_space.shape[0],))
 
    def actions(self):
        return dict(type='float', num_values=self.action_space.shape[0])
        
    def reset(self, mode='train'):
        """
        Re-initialize environment and return first observation
        mode: 'train', 'val', 'test', 'manual
        """

        # 1. Wähle das Jahr
        if mode == 'train':
            chosen_year = np.random.choice(self.train_years)
        elif mode == 'val':
            chosen_year = np.random.choice(self.val_years)
        elif mode == 'test':
            chosen_year = self.test_year
        elif mode == 'manual':
            chosen_year = self.manual_year
        else:
            raise ValueError("Mode must be 'train', 'val', or 'test' or 'manual")

        self.chosen_year = chosen_year

        print("Starting new season in year ", self.chosen_year)
        
        # 3. Irrigation Cap
        if isinstance(self.max_irr_season, list):
            if isinstance(self.max_irr_season[0], list):
                self.chosen_max_irr_season = float(np.random.choice(self.max_irr_season[0]))
            else:
                self.chosen_max_irr_season = float(np.random.randint(self.max_irr_season[0], self.max_irr_season[1]))
        else:
            self.chosen_max_irr_season = self.max_irr_season * 1.

        # 4. AquaCrop Model Setup
        month = self.planting_month
        day = self.planting_day
        
        self.model = AquaCropModel(
            f'{self.chosen_year}/{month}/{day}',
            f'{self.chosen_year}/12/31',
            self.weather_df, self.soil, self.crop,
            IrrMngt=IrrMngtClass(IrrMethod=5, MaxIrrSeason=self.chosen_max_irr_season, MaxIrr=34),
            InitWC=self.init_wc,
            CO2conc=self.CO2conc
        )
        self.model.initialize()

        if not self.include_rain:
            self.model.weather[:,2] = 0

        # Optionaler Start-Day-Shift
        if self.dayshift:
            dayshift = np.random.randint(1, self.dayshift + 1)
            self.model.step(dayshift)

        self.irr_sched = []

        return self.get_obs(self.model.InitCond)

    def calc_Wr(self, th, dz, Zroot):
        Wr = 0.0
        depth_acc = 0.0
        for theta, d in zip(th, dz):
            if depth_acc + d <= Zroot:
                # komplette Schicht innerhalb der Wurzelzone
                Wr += theta * d * 1000  # mm
            elif depth_acc < Zroot:
                # teilweise durchwurzelt
                Wr += theta * (Zroot - depth_acc) * 1000
                break
            depth_acc += d
        return Wr


    def get_obs(self,InitCond):
        """
        package variables from InitCond into a numpy array
        and return as observation
        """

        # calculate relative depletion
        if InitCond.TAW>0:
            dep = InitCond.Depletion/InitCond.TAW
        else:
            dep=0

        # calculate mean daily precipitation and ETo from last 7 days
        start = max(0,self.model.ClockStruct.TimeStepCounter -7)
        end = self.model.ClockStruct.TimeStepCounter
        forecast1 = self.model.weather[start:end,2:4].mean(axis=0).flatten()

        # calculate sum of daily precipitation and ETo for whole season so far
        start2 = max(0,self.model.ClockStruct.TimeStepCounter -InitCond.DAP)
        forecastsum = self.model.weather[start2:end,2:4].sum(axis=0).flatten()

        #  yesterday precipitation and ETo and irr
        start2 = max(0,self.model.ClockStruct.TimeStepCounter-1)
        forecast_lag1 = self.model.weather[start2:end,2:4].flatten()

        # calculate mean daily precipitation and ETo for next N days
        start = self.model.ClockStruct.TimeStepCounter
        end = start+self.forecast_lead_time
        forecast2 = self.model.weather[start:end,2:4].mean(axis=0).flatten()
        
        # state 

        # month and day
        month = (self.model.ClockStruct.TimeSpan[self.model.ClockStruct.TimeStepCounter]).month
        day = (self.model.ClockStruct.TimeSpan[self.model.ClockStruct.TimeStepCounter]).day
        
        # concatenate all weather variables

        if self.observation_set in ['default']:
            forecast = np.concatenate([forecast1,forecastsum,forecast_lag1]).flatten()
        
        elif self.observation_set=='forecast':
            forecast = np.concatenate([forecast1,forecastsum,forecast_lag1,forecast2,]).flatten()

        # put growth stage in one-hot format

        gs = np.clip(int(self.model.InitCond.GrowthStage)-1,0,4)
        gs_1h = np.zeros(4)
        gs_1h[gs]=1

        # create observation array
        
        Wr = self.calc_Wr(InitCond.th, self.soil.profile.dz, InitCond.Zroot)

        print("Root zone water content (mm): ", Wr)
        
        obs = np.array([
                    dep,  # root-zone depletion
                    self.max_irr_season - InitCond.IrrCum,  # remaining irrigation
                    Wr, 
                    InitCond.GrowthStage,
                    InitCond.Ksw.Exp,
                ], dtype=np.float32).reshape(-1)
        
        return obs
        
    def step(self,action):
        """
        Take in agents action choice

        apply irrigation depth on following day

        simulate N days until next irrigation decision point

        calculate and return profit at end of season

        """
        #print("Action taken: ", action)
        
        # if choosing discrete depths

        if self.action_set in ['depth_discreet']:

            depth = self.action_depths[int(action)]

            self.model.ParamStruct.IrrMngt.depth = depth

        # if making banry yes/no irrigation decisions

        elif self.action_set in ['binary']:

            if action == 1:
                depth = self.max_irr #apply max irr
            else:
                depth=0
            
            self.model.ParamStruct.IrrMngt.depth = depth

        # if spefiying depth from continuous range

        elif self.action_set in ['depth']:

            depth=np.clip(action[0],0,self.max_irr)
            self.model.ParamStruct.IrrMngt.depth = depth

        # if deciding on soil-moisture targets

        elif self.action_set=='smt4':

            new_smt=np.ones(4)*(action+1)*50


        start_day = self.model.InitCond.DAP

        for i in range(self.days_to_irr):

            # apply depth next day, and no more events till next decision
            
            if self.action_set in ['depth_discreet','binary','depth']:
                self.irr_sched.append(self.model.ParamStruct.IrrMngt.depth)
                self.model.step()
                self.model.ParamStruct.IrrMngt.depth = 0
            
            # if specifying soil-moisture target, 
            # irrigate if root zone soil moisture content
            # drops below threshold

            elif self.action_set=='smt4':

                if self.model.InitCond.TAW>0:
                    dep = self.model.InitCond.Depletion/self.model.InitCond.TAW
                else:
                    dep=0

                gs = int(self.model.InitCond.GrowthStage)-1
                if gs<0 or gs>3:
                    depth=0
                else:
                    if 1-dep< (new_smt[gs])/100:
                        depth = np.clip(self.model.InitCond.Depletion,0,self.max_irr)
                    else:
                        depth=0
    
                self.model.ParamStruct.IrrMngt.depth = depth
                self.irr_sched.append(self.model.ParamStruct.IrrMngt.depth)

                self.model.step()


            # termination conditions

            if self.model.ClockStruct.ModelTermination is True:
                break

            now_day = self.model.InitCond.DAP
            if (now_day >0) and (now_day<start_day):
                # end of season
                break
 
 
        done = self.model.ClockStruct.ModelTermination
        
        reward = 0
 
        next_obs = self.get_obs(self.model.InitCond)
 
        if done:
        
            self.tsteps+=1

            if self.chosen_year == 1980:
                print("Harvested season in year ", self.chosen_year)
                print("Harvest Date: ", self.model.Outputs.Final['Harvest Date (YYYY/MM/DD)'].values[0])
                print("Yield: ", self.model.Outputs.Final['Yield (tonne/ha)'].values[0])
                print("Seasonal Irrigation: ", self.model.Outputs.Final['Seasonal irrigation (mm)'].values[0])

                yields = self.model.Outputs.Final['Yield (tonne/ha)'].values[0]

                file_name = f"{self.max_irr_season}_{self.chosen_year}_{self.tsteps}_{yields:.2f}"

                self.model.Outputs.Flux.to_csv(f'logs/{file_name}_flux.csv')
                self.model.Outputs.Final.to_csv(f'logs/{file_name}_yields.csv')

            # calculate profit 
            end_reward = (self.CROP_PRICE*self.model.Outputs.Final['Yield (tonne/ha)'].mean()
                        - self.IRRIGATION_COST*self.model.Outputs.Final['Seasonal irrigation (mm)'].mean()
                        - self.FIXED_COST )

            
            self.reward=end_reward
 
            # keep track of best rewards in each season
            #rew = end_reward - self.best[self.chosen-1] 
            #if rew>0:
            #    self.best[self.chosen-1]=end_reward
            #if self.tsteps%100==0:
            #    self.total_best=self.best*1
            #    # print(self.chosen,self.tsteps,self.best[:self.year2].mean())

            # scale reward
            reward=end_reward
 
 
        return next_obs,reward,done,dict()
 
 
    
    def get_mean_std(self,num_reps):
        """
        Function to get the mean and std of observations in an environment
 
        *Arguments:*
 
        `env`: `Env` : chosen environment
        `num_reps`: `int` : number of repetitions
 
        *Returns:*
 
        `mean`: `float` : mean of observations
        `std`: `float` : std of observations
 
        """
        self.mean=0
        self.std=1
        obs=[]
        for i in range(num_reps):
            self.reset()
 
            d=False
            while not d:
 
                ob,r,d,_=self.step(np.random.choice([0,1],p=[0.9,0.1]))
                # ob,r,d,_=self.step(-0.5)
                # ob,r,d,_=self.step(np.random.choice([-1.,0.],p=[0.9,0.1]))
                obs.append(ob)
 
        obs=np.vstack(obs)
 
        mean=obs.mean(axis=0)
 
        std=obs.std(axis=0)
        std[std==0]=1
 
        self.mean=mean
        self.std=std