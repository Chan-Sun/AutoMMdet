# -*- coding: utf-8 -*-
import math 
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn import tree
from HPO.hyperopt_yxc import fmin,space_eval,hp,pyll
from scipy.stats import norm,stats
    
class AutoHPO():
    def __init__(self,space,trials,trials_best):
        self.space = space
        self.trials = trials            
        self._gama = 1/4
        self.trials_best = trials_best
    
    def choice_get_value(self,hyper_name,value):
        orig_choice=[]
        space_temp={hyper_name:self.space[hyper_name]}
        for i in range(value):
            temp={hyper_name:i}
            try:
                orig_choice.append(space_eval(space_temp,temp)[hyper_name])
            except:
                break
        return orig_choice
    
    def hyper_eval(self, hp_assignment):
        temp = pyll.as_apply(self.space)
        nodes = pyll.toposort(temp)
        memo = {}
        for node in nodes:
            if node.name == 'hyperopt_param':#判断是否为超参数Appy
                label = node.arg['label'].eval() 
                memo[label] = node 
        return memo

    def trial_for_choice(self,unique,hyper_name,total_hyper):                     
        del_v=[]  
        for i in range(len(unique)):
            new_v=i
            del_v.append(new_v)
            temp_trial=total_hyper[total_hyper[hyper_name] == unique[i]]
            temp_idx=temp_trial.index.tolist()
            for j in range(len(temp_trial)):
                idx=temp_idx[j]
                self.trials=self.trials.update_trial(hyper_name,idx,new_v) #idx控制避免多换
                
        tdel_v=list(set(del_v).difference(set(unique)))  
        tunique=list(set(unique).difference(set(del_v)))   
        for i in range(len(tdel_v)):
            new_v=tdel_v[i]
            temp_trial=total_hyper[total_hyper[hyper_name] == new_v]
            temp_idx=temp_trial.index.tolist() 
            for j in range(len(temp_trial)):
                idx=temp_idx[j]
                self.trials=self.trials.update_trial(hyper_name,idx,int(tunique[i])) 
                
        self.trials_best[hyper_name]=0 
        return self
    
    #TODO norm.fit改进  可能随着次数增加，会慢慢收敛
    def update_range(self,hyper_name,X,hyper_param):
        L=X.shape[0]
        X_best=X.iloc[0:int(L*self._gama)]
        X_best=X_best.dropna(axis=0, how='any')
        if len(X_best)<3:
            return self
        
        hyper_param=(hyper_param.arg['obj']).arg        
        if X_best.all():
            if 'low' in hyper_param:#uniform or quniform
                mu, std = norm.fit(X_best)               
                hyper_param['low']._obj=max(hyper_param['low']._obj,min(mu-2*std,X_best.iloc[0])) 
                hyper_param['high']._obj=min(hyper_param['high']._obj,max(mu+2*std,X_best.iloc[0])) 
            else:#choice型无嵌套
                unique=X_best.unique().tolist()
                orig_upper=hyper_param['upper']._obj
                if len(unique) <= orig_upper:         
                    orig_choice=self.choice_get_value(hyper_name,orig_upper)
                    
                    if orig_choice!=[]: 
                        new_choice=[]
                        for i in range(len(unique)):
                            new_choice.append(orig_choice[int(unique[i])])
                      
                        X=pd.DataFrame(X) 
                        self.trial_for_choice(unique,hyper_name,X)
                        self.space[hyper_name] = hp.choice(hyper_name,new_choice)
               
        return self
    
    
    def update_space(self,trials_total,evals):
        total_hyper,y = self._result_split(trials_total)
        total_hyper_name=total_hyper.columns.values.tolist() 
        hyper_apply = self.hyper_eval(total_hyper_name)      
        
        firstl_name=[]
        for key in self.space:
           firstl_name.append(key)         
        fea_combine = self._rank_combine(firstl_name,total_hyper[firstl_name],y)
        
        for i in range(len(firstl_name)-1):
            tune_name = firstl_name[i]
            if tune_name in hyper_apply:
                if fea_combine.iloc[i][0]>0.1 and fea_combine.iloc[i][1]>0.1 and fea_combine.iloc[i][2]<0.1:    
                    self.space[tune_name] = space_eval(self.space, self.trials_best)[tune_name]#不相关则赋最优
                else:
                    self.update_range(tune_name,total_hyper[tune_name],hyper_apply[tune_name])#相关则缩减

        if 'optimer' in hyper_apply:
            orig_value=self.trials_best['optimer']
            if fea_combine.iloc[-1][0]>0.1 and fea_combine.iloc[-1][1]>0.1 and fea_combine.iloc[-1][2]<0.1:
                #不相关，仅对trial_best内部赋值 每次更新  
                temp_dict=space_eval(self.space,self.trials_best)['optimer']
                secdl_name=temp_dict['optimer']
                secdl_subname=temp_dict[secdl_name]
                secdl_dict={}
                for key in secdl_subname:
                    self.update_range(key,total_hyper[key],hyper_apply[key])
                    hyper_param=hyper_apply[key]
                    low=((hyper_param.arg['obj']).arg)['low']._obj
                    high=((hyper_param.arg['obj']).arg)['high']._obj
                    secdl_dict[key]=hp.uniform(key,low,high)  
                
                new_value=0
                if  new_value!=orig_value:
                    self.trial_for_choice([orig_value],'optimer',total_hyper)
                self.space['optimer']=hp.choice('optimer', [{'optimer':secdl_name,secdl_name:secdl_dict}])
                
            else:
                #相关，对X_best内部进行优化  如果相关，也是要进行内部优化的
                new_list=[]
                L=total_hyper.shape[0]
                trials_best=total_hyper.iloc[0:int(L*self._gama)]
                unique=trials_best['optimer'].unique().tolist()
                
                if ((hyper_apply['optimer'].arg['obj']).arg)['upper']._obj<len(unique):
                    unique=[self.trials_best['optimer']]
                    
                unique_name=[]
                for i in range(len(unique)):
                    temp_trial=trials_best[trials_best['optimer'] == unique[i]].iloc[0]
                    temp_dict=(temp_trial.dropna(axis=0, how='any')).to_dict() 
                    
                    for key in self.space:
                        if key != 'optimer':
                            if isinstance(self.space[key], float) or isinstance(self.space[key], str):
                                temp_dict[key]=self.space[key]
                            else:
                                temp_dict[key]=self.trials_best[key]
                    
                    temp_dict=space_eval(self.space,temp_dict)['optimer']
                    secdl_name=temp_dict['optimer']
                    unique_name.append(secdl_name)
                    secdl_subname=temp_dict[secdl_name]
                    
                    secdl_dict={}
                    for key in secdl_subname: 
                        self.update_range(key,total_hyper[key],hyper_apply[key])
                        hyper_param=hyper_apply[key]
                        low=((hyper_param.arg['obj']).arg)['low']._obj
                        high=((hyper_param.arg['obj']).arg)['high']._obj
                        secdl_dict[key]=hp.uniform(key,low,high)                
                    new_list.append({'optimer':secdl_name,secdl_name:secdl_dict})
             
                self.trial_for_choice(unique,'optimer',total_hyper)
                self.space['optimer']=hp.choice('optimer', new_list)  
                
        return self.space               

    def _result_not_null(self,result,evals):       
        for key in list(result.keys()):  
            if len(result[key]) != evals: 
                del result[key]
        result = pd.DataFrame(result) 
        return result
    
    def _dict_to_dataframe(self,result,evals):
        if isinstance(result,dict):
            for key in list(result.keys()):  
                if len(result[key]) != evals: 
                    del result[key]
            result = pd.DataFrame(result) 
        return result
           
    def _result_split(self,result):        
        result = result.sort_values(by="loss" , ascending=True)  
        
        len_result = result.shape[0]  
        y = result['loss']
        y.iloc[0:int(len_result*self._gama)] = 1
        y[int(len_result*self._gama):len_result] = 0  
        
        hyper_parameter = result.drop(['loss'],axis=1)  
        return hyper_parameter,y

    def _rank_combine(self,firstl_name,X,y):
        #DTree
        m=tree.DecisionTreeClassifier()
        m.fit(X,y)
        fea_im=pd.DataFrame(m.feature_importances_,columns=['fea_im'])

        #p_value   
        k_best = SelectKBest(f_classif,k='all')
        k_best.fit_transform(X, y)
        p_values =pd.DataFrame(k_best.pvalues_,columns=['p_value'])
        
        #corr
        corr_p=pd.DataFrame(columns=['corr_p'])
        for i in range(len(firstl_name)):
            corr_p.loc[i,'corr_p']=stats.pearsonr(X[firstl_name[i]],y)[1]
            
        fea_combine = pd.concat([corr_p,p_values, fea_im], axis=1)
        return fea_combine
    


def space_apply(space, hp_assignment):
    temp = pyll.as_apply(space)
    nodes = pyll.toposort(temp)
    memo = {}
    for node in nodes:
        if node.name == 'hyperopt_param':
            label = node.arg['label'].eval() 
            memo[label] = node 
    return memo
    
def space_save_df(space,evals):
    total_hyper_name=[key for key in space]
    hyper_apply = space_apply(space,total_hyper_name)
    i=0
    space_df=pd.DataFrame(columns=['evals','name', 'low', 'high', 'upper'])
    space_df.loc[i,'evals']=evals
    for key in hyper_apply:
        i=i+1
        space_df.loc[i,'name']=key
        hyper_param=hyper_apply[key]
        hyper_param=(hyper_param.arg['obj']).arg
        if 'low' in hyper_param:      
            space_df.loc[i,'low']=hyper_param['low']._obj
            space_df.loc[i,'high']=hyper_param['high']._obj
        else:
            space_df.loc[i,'upper']=hyper_param['upper']._obj    
    return space_df


def fmin_hyperp_reduce(fn,space, algo, max_evals, trials):#在类外实例化类
    
    space_save=pd.DataFrame(columns=['evals','name', 'low', 'high', 'upper'])

    eta=2
    drop_number = int(math.log(max_evals, eta)) 
    drop_number = [i for i in range(1,drop_number+1)]
    drop_number.append(100000)  
    for i in drop_number:         
        evals = int(max_evals*(1-math.pow(1/eta,i))) 
        trials_best = fmin(fn=fn, space=space, algo=algo, max_evals=evals,trials=trials) 
        trials_total=trials_to_df(trials)
        fill_trials(trials_total,space)    #便于算第一层节点的重要性，因为不能存在nan         
        if evals == max_evals:
            temp=space_save_df(space,evals)
            space_save=space_save.append(temp, ignore_index=True) 
            break        
        
        auto=AutoHPO(space,trials,trials_best)            
        space=auto.update_space(trials_total,evals)
        
        temp=space_save_df(space,evals) 
        space_save=space_save.append(temp, ignore_index=True)  
        
    return space_eval(space, trials_best),trials_total,space_save 

def fmin_raw(fn,space, algo, max_evals, trials):
    trials_best = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals,trials=trials) 
    min_loss = trials.best_trial['result']['loss']
    trials_results = trials_to_df(trials)
    return space_eval(space, trials_best),trials_results,trials_results

def trials_to_df(trials):
        candidates = [t for t in trials if t['result']['status'] == 'ok'] 
        rval = pd.DataFrame({'index':[x for x in range(len(candidates))]})
        for i in range(len(candidates)):
            rval.loc[i,'loss']=candidates[i]['result']['loss']
            vals = candidates[i]['misc']['vals']              
            for k, v in list(vals.items()):
                if v:
                    rval.loc[i,k] = v[0]
        rval=rval.drop(['index'],axis=1)
        return rval
    
def fill_trials(trials_total,space):        
    trials_fill=trials_total.fillna('9999999')
    for key in space.keys():
        if trials_total[key].isnull().any() == True :
            null_idx= min(trials_fill[trials_fill[key] == '9999999'].index.tolist())           
            loss_temp = trials_fill['loss'][0:null_idx]
            loss_idx = loss_temp[loss_temp==min(loss_temp[0:null_idx])].index         
            key_best=trials_total.loc[loss_idx,key].values.tolist()          
            trials_total[key]=trials_total[key].fillna(key_best[0])