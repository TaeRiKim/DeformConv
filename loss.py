import torch
    
def MainLoss(loss0, loss1, loss2, alpha):
    mainloss = (1.0 - alpha)*loss0 + alpha*loss1/2 + alpha*loss2/2
    return mainloss

def AuxLoss(loss0, loss1, loss2, betha):
    auxloss = (1.0 - betha)*loss0 + betha*loss1/2 + betha*loss2/2
    return auxloss    

def HybridLoss(mainloss, auxloss, gamma):
    hybridloss = mainloss + gamma*auxloss
    return hybridloss

    