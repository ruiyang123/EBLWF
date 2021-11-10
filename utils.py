import numpy as np
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.nn import functional as F
from torchvision import transforms
import copy
import data

###################
## Loss function ##
###################


def cos_sim_loss(f1,f2,y1,y2):
    f1 = torch.flatten(f1, 1)
    f2 = torch.flatten(f2, 1)
    target = []
    for i in y1==y2[0]:
        if i:
            target.append(1)
        else:
            target.append(-1)
    target = torch.tensor(target, dtype=torch.int8).cuda()

    loss = F.cosine_embedding_loss(f1,f2,target,margin=0.5,reduction="mean")
    return loss



def entropy_info(feature):
    if len(feature.size())==4:
        size0 = feature.size()[0]
        size1 = feature.size()[1]
        size2 = feature.size()[2]
        size3 = feature.size()[3]
        new_fea = feature.view(size0,size1,-1,1)
        new_fea = F.softmax(new_fea,dim=2)
        new_fea = new_fea * torch.log(new_fea)
        new_fea = torch.sum(new_fea,dim=2,keepdim=True)
        return 1 - new_fea/np.log(1/size2/size3)
    else:
        return torch.ones_like(feature)

def pod(list_attentions_a, list_attentions_b, collapse_channels="spatial", normalize=True,adjust=True):
    # attention map : b_size * n_channels * width * height
    assert len(list_attentions_a) == len(list_attentions_b)
    loss = torch.tensor(0.).to(list_attentions_a[0].device)
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        assert a.shape == b.shape, (a.shape, b.shape)
        shapes = a.size()
        if adjust:
            adjust_weight = entropy_info(b).detach()
        else:
            adjust_weight = 0
        a = torch.pow(a, 2)
        b = torch.pow(b, 2)

        if collapse_channels == "channels":
            a = a.sum(dim=1).view(a.shape[0], -1)
            b = b.sum(dim=1).view(b.shape[0], -1)
        elif collapse_channels == "width":
            a = a.sum(dim=2).view(a.shape[0], -1)
            b = b.sum(dim=2).view(b.shape[0], -1)
        elif collapse_channels == "height":
            a = a.sum(dim=3).view(a.shape[0], -1)
            b = b.sum(dim=3).view(b.shape[0], -1)
        elif collapse_channels == "spatial":
            a_h = a.sum(dim=3).view(a.shape[0], -1)
            b_h = b.sum(dim=3).view(b.shape[0], -1)
            a_w = a.sum(dim=2).view(a.shape[0], -1)
            b_w = b.sum(dim=2).view(b.shape[0], -1)
            a = torch.cat([a_h, a_w], dim=-1)
            b = torch.cat([b_h, b_w], dim=-1)

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        a = a.view((shapes[0],shapes[1],-1,1))
        b = b.view((shapes[0],shapes[1],-1,1))

        layer_loss = torch.norm(a-b,dim=2,keepdim=True)


        layer_loss = layer_loss * adjust_weight
        layer_loss = torch.mean(layer_loss)




        #layer_loss = torch.mean(torch.frobenius_norm(a - b), dim=-1)
        loss += layer_loss
    return loss / len(list_attentions_a)



def adjust_loss_fn_kd(scores, target_scores, T=2):
    device = scores.device

    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)

    task_num = target_scores.size(1)
    p_target = F.softmax(target_scores.detach(), dim=1)
    log_p_target = torch.log(p_target.detach())
    adjust_weight = torch.sum(log_p_target*p_target,dim=1) / np.log(1./task_num)
    #print(adjust_weight)

    # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    n = scores.size(1)
    if n>target_scores.size(1):
        n_batch = scores.size(0)
        zeros_to_add = torch.zeros(n_batch, n-target_scores.size(1))
        zeros_to_add = zeros_to_add.to(device)
        targets_norm = torch.cat([targets_norm.detach(), zeros_to_add], dim=1)

    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    KD_loss_unnorm = -(targets_norm * log_scores_norm)
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1) * (1-adjust_weight)                     #--> sum over classes
    KD_loss_unnorm = KD_loss_unnorm.mean()                          #--> average over batch

    # normalize
    KD_loss = KD_loss_unnorm * T**2

    return KD_loss





def loss_fn_kd(scores, target_scores, T=2.):
    """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].

    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""

    device = scores.device

    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)

    # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    n = scores.size(1)
    if n>target_scores.size(1):
        n_batch = scores.size(0)
        zeros_to_add = torch.zeros(n_batch, n-target_scores.size(1))
        zeros_to_add = zeros_to_add.to(device)
        targets_norm = torch.cat([targets_norm.detach(), zeros_to_add], dim=1)

    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    KD_loss_unnorm = -(targets_norm * log_scores_norm)
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)                      #--> sum over classes
    KD_loss_unnorm = KD_loss_unnorm.mean()                          #--> average over batch

    # normalize
    KD_loss = KD_loss_unnorm * T**2

    return KD_loss


def loss_fn_kd_binary(scores, target_scores, T=2.):
    """Compute binary knowledge-distillation (KD) loss given [scores] and [target_scores].

    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""

    device = scores.device

    scores_norm = torch.sigmoid(scores / T)
    targets_norm = torch.sigmoid(target_scores / T)

    # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    n = scores.size(1)
    if n>target_scores.size(1):
        n_batch = scores.size(0)
        zeros_to_add = torch.zeros(n_batch, n-target_scores.size(1))
        zeros_to_add = zeros_to_add.to(device)
        targets_norm = torch.cat([targets_norm, zeros_to_add], dim=1)

    # Calculate distillation loss
    KD_loss_unnorm = -( targets_norm * torch.log(scores_norm) + (1-targets_norm) * torch.log(1-scores_norm) )
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)      #--> sum over classes
    KD_loss_unnorm = KD_loss_unnorm.mean()          #--> average over batch

    # normalize
    KD_loss = KD_loss_unnorm * T**2

    return KD_loss


##-------------------------------------------------------------------------------------------------------------------##


#############################
## Data-handling functions ##
#############################

def get_data_loader(dataset, batch_size, cuda=False, collate_fn=None, drop_last=False, augment=False,shuffle=True):
    '''Return <DataLoader>-object for the provided <DataSet>-object [dataset].'''

    # If requested, make copy of original dataset to add augmenting transform (without altering original dataset)
    if augment:
        dataset_ = copy.deepcopy(dataset)
        dataset_.transform = transforms.Compose([dataset.transform, *data.AVAILABLE_TRANSFORMS['augment']])
    else:
        dataset_ = dataset
    #print('batch size: '+str(batch_size))
    #print('dataset: '+ str(len(dataset)))
    # Create and return the <DataLoader>-object
    loader = DataLoader(
        dataset_, batch_size=batch_size, shuffle=shuffle,
        collate_fn=(collate_fn or default_collate), drop_last=drop_last,
        **({'num_workers': 0, 'pin_memory': True} if cuda else {})
    )

    return loader

def label_squeezing_collate_fn(batch):
    x, y = default_collate(batch)
    return x, y.long().squeeze()


def to_one_hot(y, classes):
    '''Convert a nd-array with integers [y] to a 2D "one-hot" tensor.'''
    c = np.zeros(shape=[len(y), classes], dtype='float32')
    c[range(len(y)), y] = 1.
    c = torch.from_numpy(c)
    return c


##-------------------------------------------------------------------------------------------------------------------##

##########################################
## Object-saving and -loading functions ##
##########################################

def save_object(object, path):
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)

def load_object(path):
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)



##-------------------------------------------------------------------------------------------------------------------##

################################
## Model-inspection functions ##
################################

def count_parameters(model, verbose=True):
    '''Count number of parameters, print to screen.'''
    total_params = learnable_params = fixed_params = 0
    for param in model.parameters():
        n_params = index_dims = 0
        for dim in param.size():
            n_params = dim if index_dims==0 else n_params*dim
            index_dims += 1
        total_params += n_params
        if param.requires_grad:
            learnable_params += n_params
        else:
            fixed_params += n_params
    if verbose:
        print("--> this network has {} parameters (~{} million)"
              .format(total_params, round(total_params / 1000000, 1)))
        print("      of which: - learnable: {} (~{} million)".format(learnable_params,
                                                                     round(learnable_params / 1000000, 1)))
        print("                - fixed: {} (~{} million)".format(fixed_params, round(fixed_params / 1000000, 1)))
    return total_params, learnable_params, fixed_params


def print_model_info(model, title="MODEL"):
    '''Print information on [model] onto the screen.'''
    print("Model-name: \"" + model.name + "\"")
    print(40*"-" + title + 40*"-")
    print(model)
    print(90*"-")
    _ = count_parameters(model)
    print(90*"-" + "\n\n")



##-------------------------------------------------------------------------------------------------------------------##

#################################
## Custom-written "nn-Modules" ##
#################################


class Identity(nn.Module):
    '''A nn-module to simply pass on the input data.'''
    def forward(self, x):
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '()'
        return tmpstr


class Reshape(nn.Module):
    '''A nn-module to reshape a tensor to a 4-dim "image"-tensor with [image_channels] channels.'''
    def __init__(self, image_channels):
        super().__init__()
        self.image_channels = image_channels

    def forward(self, x):
        batch_size = x.size(0)   # first dimenstion should be batch-dimension.
        image_size = int(np.sqrt(x.nelement() / (batch_size*self.image_channels)))
        return x.view(batch_size, self.image_channels, image_size, image_size)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '(channels = {})'.format(self.image_channels)
        return tmpstr


class ToImage(nn.Module):
    '''Reshape input units to image with pixel-values between 0 and 1.

    Input:  [batch_size] x [in_units] tensor
    Output: [batch_size] x [image_channels] x [image_size] x [image_size] tensor'''

    def __init__(self, image_channels=1):
        super().__init__()
        # reshape to 4D-tensor
        self.reshape = Reshape(image_channels=image_channels)
        # put through sigmoid-nonlinearity
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.reshape(x)
        x = self.sigmoid(x)
        return x

    def image_size(self, in_units):
        '''Given the number of units fed in, return the size of the target image.'''
        image_size = np.sqrt(in_units/self.image_channels)
        return image_size


class Flatten(nn.Module):
    '''A nn-module to flatten a multi-dimensional tensor to 2-dim tensor.'''
    def forward(self, x):
        batch_size = x.size(0)   # first dimenstion should be batch-dimension.
        return x.view(batch_size, -1)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '()'
        return tmpstr


##-------------------------------------------------------------------------------------------------------------------##

# We only calculate the diagonal elements of the hessian
def logistic_hessian(f):
    f = f[:, :]
    pi = torch.sigmoid(f)
    return pi*(1-pi)


def softmax_hessian(f):
    s = F.softmax(f, dim=-1)
    return s - s*s


# Calculate the full softmax hessian
def full_softmax_hessian(f):
    s = F.softmax(f, dim=-1)
    e = torch.eye(s.shape[-1], dtype=s.dtype, device=s.device)
    return s[:, :, None]*e[None, :, :] - s[:, :, None]*s[:, None, :]


# Select memorable points ordered by their lambda values (descending=True picks most important points)
def select_memorable_points(dataloader, model, num_points=10, num_classes=2,
                            use_cuda=False, label_set=None, descending=True):
    memorable_points = {}
    scores = {}
    num_points_per_class = int(num_points/num_classes)
    for i, dt in enumerate(dataloader):
        data, target = dt
        if use_cuda:
            data_in = data.cuda()
        else:
            data_in = data
        if label_set == None:
            f = model.forward(data_in)
        else:
            f = model.forward(data_in, label_set)
        if f.shape[-1] > 1:
            lamb = softmax_hessian(f)
            if use_cuda:
                lamb = lamb.cpu()
            lamb = torch.sum(lamb, dim=-1)
            lamb = lamb.detach()
        else:
            lamb = logistic_hessian(f)
            if use_cuda:
                lamb = lamb.cpu()
            lamb = torch.squeeze(lamb, dim=-1)
            lamb = lamb.detach()
        for cid in range(num_classes):
            p_c = data[target == cid]
            if len(p_c) > 0:
                s_c = lamb[target == cid]
                if len(s_c) > 0:
                    if cid not in memorable_points:
                        memorable_points[cid] = p_c
                        scores[cid] = s_c
                    else:
                        memorable_points[cid] = torch.cat([memorable_points[cid], p_c], dim=0)
                        scores[cid] = torch.cat([scores[cid], s_c], dim=0)
                        if len(memorable_points[cid]) > num_points_per_class:
                            _, indices = scores[cid].sort(descending=descending)
                            memorable_points[cid] = memorable_points[cid][indices[:num_points_per_class]]
                            scores[cid] = scores[cid][indices[:num_points_per_class]]
    r_points = []
    r_labels = []
    for cid in range(num_classes):
        r_points.append(memorable_points[cid])
        r_labels.append(torch.ones(memorable_points[cid].shape[0], dtype=torch.long,
                                   device=memorable_points[cid].device)*cid)
    return [torch.cat(r_points, dim=0), torch.cat(r_labels, dim=0)]


# Randomly select some points as memory
def random_memorable_points(dataset, num_points, num_classes):
    memorable_points = {}
    num_points_per_class = int(num_points/num_classes)
    exact_num_points = num_points_per_class*num_classes
    idx_list = torch.randperm(len(dataset))
    select_points_num = 0
    for idx in range(len(idx_list)):
        data, label = dataset[idx_list[idx]]
        cid = label.item() if isinstance(label, torch.Tensor) else label
        if cid in memorable_points:
            if len(memorable_points[cid]) < num_points_per_class:
                memorable_points[cid].append(data)
                select_points_num += 1
        else:
            memorable_points[cid] = [data]
            select_points_num += 1
        if select_points_num >= exact_num_points:
            break
    r_points = []
    r_labels = []
    for cid in range(num_classes):
        r_points.append(torch.stack(memorable_points[cid], dim=0))
        r_labels.append(torch.ones(len(memorable_points[cid]), dtype=torch.long,
                                   device=r_points[cid].device)*cid)
    return [torch.cat(r_points, dim=0), torch.cat(r_labels, dim=0)]


# Update the fisher matrix after training on a task
def update_fisher(dataloader, model, opt, label_set=None, use_cuda=False):
    model.eval()
    for data, label in dataloader:
        if use_cuda:
            data = data.cuda()
        def closure():
            opt.zero_grad()
            if label_set == None:
                logits = model.forward(data)
            else:
                logits = model.forward(data, label_set)
            return logits
        opt.update_fisher(closure)


def save(opt, memorable_points, path):
    torch.save({
        'mu': opt.state['mu'],
        'fisher': opt.state['fisher'],
        'memorable_points': memorable_points
    }, path)


def load(opt, path):
    checkpoint = torch.load(path)
    opt.state['mu'] = checkpoint['mu']
    opt.state['fisher'] = checkpoint['fisher']
    return checkpoint['memorable_points']


def softmax_predictive_accuracy(logits_list, y, ret_loss = False):
    probs_list = [F.log_softmax(logits, dim=1) for logits in logits_list]
    probs_tensor = torch.stack(probs_list, dim = 2)
    probs = torch.mean(probs_tensor, dim=2)
    if ret_loss:
        loss = F.nll_loss(probs, y, reduction='sum').item()
    _, pred_class = torch.max(probs, 1)
    correct = pred_class.eq(y.view_as(pred_class)).sum().item()
    if ret_loss:
        return correct, loss
    return correct



