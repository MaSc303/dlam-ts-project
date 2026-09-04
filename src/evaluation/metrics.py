import numpy as np

def wape(target, prediction):
    target, prediction = np.asarray(target), np.asarray(prediction)
    return np.abs(target - prediction).sum() / np.abs(target).sum()

def all_metrics(y,p):
    y,p=np.asarray(y),np.asarray(p)
    e=y-p
    return {"mae":np.abs(e).mean(),"mse":np.square(e).mean(),"rmse":np.sqrt(np.square(e).mean()),"mape":np.mean(np.abs(e)/np.maximum(np.abs(y),1e-8))*100,"smape":np.mean(2*np.abs(e)/np.maximum(np.abs(y)+np.abs(p),1e-8))*100,"wape":np.abs(e).sum()/np.abs(y).sum()*100}