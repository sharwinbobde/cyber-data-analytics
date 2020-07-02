# coding=utf-8
"""
Python module for implementing inner maximizers for robust adversarial training
(Table I in the paper)
"""
import torch
from torch.autograd import Variable
from utils.utils import or_float_tensors, xor_float_tensors, clip_tensor
import numpy as np
import random
import pyswarms as ps
from time import time


# helper function
def round_x(x, alpha=0.5):
    """
    rounds x by thresholding it according to alpha which can be a scalar or vector
    :param x:
    :param alpha: threshold parameter
    :return: a float tensor of 0s and 1s.
    """
    return (x > alpha).float()


def get_x0(x, is_sample=False):
    """
    Helper function to randomly initialize the the inner maximizer algos
    randomize such that the functionality is preserved.
    Functionality is preserved by maintaining the features present in x
    :param x: training sample
    :param is_sample: flag to sample randomly from feasible area or return just x
    :return: randomly sampled feasible version of x
    """
    if is_sample:
        rand_x = round_x(torch.rand(x.size()))
        if x.is_cuda:
            rand_x = rand_x.cuda()
        return or_float_tensors(x, rand_x)
    else:
        return x
        

def pso_loss(params, y, model, loss_fct):
    print(params.shape)
    xn_var = Variable(params, requires_grad=True)
    y_model = model(xn_var)
    loss = loss_fct(y_model, y)
    print(loss)
    exit(0)
        
def particle_swarm(x, y, model, loss_fct):
    if next(model.parameters()).is_cuda:
        x = x.cuda()
        y = y.cuda()

    x_next = get_x0(x)
    x_orig = get_x0(x)
    x_best = get_x0(x)
    y = Variable(y)
    
    pso_options = {'c1': 4, 'c2':4, 'w':0.9, 'k':5, 'p':2}
    dims = x.shape[1]
    jobs = x.shape[0]
    print('jobs, dims = ' +str(x.shape))
    optimizer =ps.binary.BinaryPSO(n_particles=22,
                                    dimensions=dims,
                                    options=pso_options)
    cost, pos = optimizer.optimize(pso_loss, iters=1000, y=y, model=model, loss_fct=loss_fct)
    if x_best.is_cuda:
        x_best = x_best.cpu()
    
    return x_best

def make_topk_map(topk, sign, shape, k=8):
    topk_values = topk[0]
    topk_indices = topk[1]
    map_ = torch.zeros(shape).cuda()
    for i in range(shape[0]):
        for j in range(k):
            index = topk_indices[i,j]
            # print(index)
            # map_[i,index] = topk_values[i, j] * sign[i,index]
            try:
                map_[i,index] = topk_values[i, j]
            except IndexError:
                print('IndexError occured :(')
                print('topk_indices = '+ str(topk_indices))
            
    hadamard_product = map_ * sign[i,index]
    return hadamard_product


def grams_topk(x, y, model, loss_fct, k=8):
    # best_x = torch.cuda.FloatTensor(x.shape[0], x.shape[1])
    # best_x.copy_(x)
    start = time()
    if next(model.parameters()).is_cuda:
        x = x.cuda()
        y = y.cuda()
    best_x = get_x0(x)
    orig_x = get_x0(x)
    y = Variable(y)
    
    counter = 0
    while k > 0.5 and counter < 128:
        # print(k)
        # forward pass
        x_var = Variable(x, requires_grad=True)
        y_pred = model(x_var)
        loss = loss_fct(y_pred, y)
        # gradient
        grad = torch.autograd.grad(loss.mean(), x_var)

        sign = torch.sign(grad[0].data)
        grad = torch.abs(grad[0].data - orig_x * grad[0].data)
        # print(grad)
        # print(torch.topk(grad, k)[0])
        # print(torch.topk(grad, k)[1])
        # print(sign)
        topk_map = make_topk_map(torch.topk(grad, int(k)), sign, shape=x.shape, k=int(k))
        # print(topk_map)
        x_ = x +  topk_map

        # forward pass
        x__var = Variable(x_, requires_grad=True)
        loss = loss_fct(model(x__var), y)
        best_x_var = Variable(best_x, requires_grad=False)
        loss_ = loss_fct(model(best_x_var), y)

        topk = loss > loss_
        # print(loss)
        # print(loss_)
        # print(topk.data)
        if torch.sum(topk.data) == 0 :
            # print('was 0 :)')
            k = k / 2
        else:
            for i in range(8):
                if topk.data[i] == 1:
                    best_x[i, :] = np.logical_or(best_x[i, :], x[i, :])
            x = get_x0(x_)
            
            if k < 512: # upper limit for k
                k = 2 * k
        counter += 1
    end = time()
    # print('topk for 1 batch took ' +str(end-start)+'s')
    return best_x

def grams_topk_plus(x, y, model, loss_fct, k=8):
    x = grams_topk(x, y, model, loss_fct, k)

    if next(model.parameters()).is_cuda:
        x = x.cuda()
        y = y.cuda()

    x_best = get_x0(x)
    x_next = get_x0(x)
    x_orig = get_x0(x)
    y_var = Variable(y)
    no_improve = 0

    while no_improve < 10:
        x_old = x_next.clone()
        xn_var = Variable(x_next, requires_grad=True)
        y_model = model(xn_var)
        loss = loss_fct(y_model, y_var)
        grads = torch.autograd.grad(loss.mean(), xn_var)[0].data

        # randomize gradient
        grads = torch.rand(grads.shape).cuda()*grads

        # topk
        signs = torch.gt(grads, 0).float()
        grads = (signs - x_next) * grads
        signs += x_orig

        #rand_vars = torch.rand(len(grads),len(grads[0]))

        kvals, kidx = grads.topk(k=random.randrange(1, 21), dim=1)
        x_next.scatter_(dim=1, index=kidx, src=signs.gather(dim=1,index=kidx))

        xn_var = Variable(x_next, requires_grad=True)
        y_model = model(xn_var)
        loss = loss_fct(y_model, y_var)

        xb_var = Variable(x_best, requires_grad=True)
        y_model = model(xb_var)
        loss_ = loss_fct(y_model, y_var)

        lower_loss = loss > loss_
        
        if lower_loss.float()[0].data.sum() == 0:
            no_improve += 1
        else:
            no_improve = 0
            x_best[lower_loss] = x_next[lower_loss]
    
    return grams_topk(x_best, y, model, loss_fct, k=8)


def dfgsm_k(
    x,
    y,
    model,
    loss_fct,
    k=25,
    epsilon=0.02,
    alpha=0.5,
    is_report_loss_diff=False,
    use_sample=False,
):
    """
    FGSM^k with deterministic rounding
    :param y:
    :param x: (tensor) feature vector
    :param model: nn model
    :param loss_fct: loss function
    :param k: num of steps
    :param epsilon: update value in each direction
    :param alpha:
    :param is_report_loss_diff:
    :param use_sample:
    :return: the adversarial version of x according to dfgsm_k (tensor)
    """
    # some book-keeping
    if next(model.parameters()).is_cuda:
        x = x.cuda()
        y = y.cuda()
    y = Variable(y)

    # compute natural loss
    loss_natural = loss_fct(model(Variable(x)), y).data

    # initialize starting point
    x_next = get_x0(x, use_sample)

    # multi-step
    for t in range(k):
        # forward pass
        x_var = Variable(x_next, requires_grad=True)
        y_model = model(x_var)
        loss = loss_fct(y_model, y)

        # compute gradient
        grad_vars = torch.autograd.grad(loss.mean(), x_var)

        # find the next sample
        x_next = x_next + epsilon * torch.sign(grad_vars[0].data)

        # projection
        x_next = clip_tensor(x_next)

    # rounding step
    x_next = round_x(x_next, alpha=alpha)

    # feasible projection
    x_next = or_float_tensors(x_next, x)

    # compute adversarial loss
    loss_adv = loss_fct(model(Variable(x_next)), y).data

    if is_report_loss_diff:
        print(
            "Natural loss (%.4f) vs Adversarial loss (%.4f), Difference: (%.4f)"
            % (
                loss_natural.mean(),
                loss_adv.mean(),
                loss_adv.mean() - loss_natural.mean(),
            )
        )

    replace_flag = (loss_adv < loss_natural).unsqueeze(1).expand_as(x_next)
    x_next[replace_flag] = x[replace_flag]

    if x_next.is_cuda:
        x_next = x_next.cpu()

    return x_next


def rfgsm_k(
    x,
    y,
    model,
    loss_fct,
    k=25,
    epsilon=0.02,
    is_report_loss_diff=False,
    use_sample=False,
):
    """
    FGSM^k with randomized rounding
    :param x: (tensor) feature vector
    :param y:
    :param model: nn model
    :param loss_fct: loss function
    :param k: num of steps
    :param epsilon: update value in each direction
    :param is_report_loss_diff:
    :param use_sample:
    :return: the adversarial version of x according to rfgsm_k (tensor)
    """
    # some book-keeping
    if next(model.parameters()).is_cuda:
        x = x.cuda()
        y = y.cuda()
    y = Variable(y)

    # compute natural loss
    loss_natural = loss_fct(model(Variable(x)), y).data

    # initialize starting point
    x_next = get_x0(x, use_sample)

    # multi-step with gradients
    for t in range(k):
        # forward pass
        x_var = Variable(x_next, requires_grad=True)
        y_model = model(x_var)
        loss = loss_fct(y_model, y)

        # compute gradient
        grad_vars = torch.autograd.grad(loss.mean(), x_var)

        # find the next sample
        x_next = x_next + epsilon * torch.sign(grad_vars[0].data)

        # projection
        x_next = clip_tensor(x_next)

    # rounding step
    alpha = torch.rand(x_next.size())
    if x_next.is_cuda:
        alpha = alpha.cuda()
    x_next = round_x(x_next, alpha=alpha)

    # feasible projection
    x_next = or_float_tensors(x_next, x)

    # compute adversarial loss
    loss_adv = loss_fct(model(Variable(x_next)), y).data

    if is_report_loss_diff:
        print(
            "Natural loss (%.4f) vs Adversarial loss (%.4f), Difference: (%.4f)"
            % (
                loss_natural.mean(),
                loss_adv.mean(),
                loss_adv.mean() - loss_natural.mean(),
            )
        )

    replace_flag = (loss_adv < loss_natural).unsqueeze(1).expand_as(x_next)
    x_next[replace_flag] = x[replace_flag]

    if x_next.is_cuda:
        x_next = x_next.cpu()

    return x_next


def bga_k(x, y, model, loss_fct, k=25, is_report_loss_diff=False, use_sample=False):
    """
    Multi-step bit gradient ascent
    :param x: (tensor) feature vector
    :param y:
    :param model: nn model
    :param loss_fct: loss function
    :param k: num of steps
    :param is_report_loss_diff:
    :param use_sample:
    :return: the adversarial version of x according to bga_k (tensor)
    """
    # some book-keeping
    sqrt_m = torch.from_numpy(np.sqrt([x.size()[1]])).float()

    if next(model.parameters()).is_cuda:
        x = x.cuda()
        y = y.cuda()
        sqrt_m = sqrt_m.cuda()

    y = Variable(y)

    # compute natural loss
    loss_natural = loss_fct(model(Variable(x)), y).data

    # keeping worst loss
    loss_worst = loss_natural.clone()
    x_worst = x.clone()

    # multi-step with gradients
    loss = None
    x_var = None
    x_next = None
    for t in range(k):
        if t == 0:
            # initialize starting point
            x_next = get_x0(x, use_sample)
        else:
            # compute gradient
            grad_vars = torch.autograd.grad(loss.mean(), x_var)
            grad_data = grad_vars[0].data

            # compute the updates
            x_update = (
                sqrt_m * (1.0 - 2.0 * x_next) * grad_data
                >= torch.norm(grad_data, 2, 1).unsqueeze(1).expand_as(x_next)
            ).float()

            # find the next sample with projection to the feasible set
            x_next = xor_float_tensors(x_update, x_next)
            x_next = or_float_tensors(x_next, x)

        # forward pass
        x_var = Variable(x_next, requires_grad=True)
        y_model = model(x_var)
        loss = loss_fct(y_model, y)

        # update worst loss and adversarial samples
        replace_flag = loss.data > loss_worst
        loss_worst[replace_flag] = loss.data[replace_flag]
        x_worst[replace_flag.unsqueeze(1).expand_as(x_worst)] = x_next[
            replace_flag.unsqueeze(1).expand_as(x_worst)
        ]

    if is_report_loss_diff:
        print(
            "Natural loss (%.4f) vs Adversarial loss (%.4f), Difference: (%.4f)"
            % (
                loss_natural.mean(),
                loss_worst.mean(),
                loss_worst.mean() - loss_natural.mean(),
            )
        )

    if x_worst.is_cuda:
        x_worst = x_worst.cpu()

    return x_worst


def bca_k(x, y, model, loss_fct, k=25, is_report_loss_diff=False, use_sample=False):
    """
    Multi-step bit coordinate ascent
    :param use_sample:
    :param is_report_loss_diff:
    :param y:
    :param x: (tensor) feature vector
    :param model: nn model
    :param loss_fct: loss function
    :param k: num of steps
    :return: the adversarial version of x according to bca_k (tensor)
    """
    if next(model.parameters()).is_cuda:
        x = x.cuda()
        y = y.cuda()

    y = Variable(y)

    # compute natural loss
    loss_natural = loss_fct(model(Variable(x)), y).data

    # keeping worst loss
    loss_worst = loss_natural.clone()
    x_worst = x.clone()

    # multi-step with gradients
    loss = None
    x_var = None
    x_next = None
    for t in range(k):
        if t == 0:
            # initialize starting point
            x_next = get_x0(x, use_sample)
        else:
            # compute gradient
            grad_vars = torch.autograd.grad(loss.mean(), x_var)
            grad_data = grad_vars[0].data

            # compute the updates (can be made more efficient than this)
            aug_grad = (1.0 - 2.0 * x_next) * grad_data
            val, _ = torch.topk(aug_grad, 1)
            x_update = (aug_grad >= val.expand_as(aug_grad)).float()

            # find the next sample with projection to the feasible set
            x_next = xor_float_tensors(x_update, x_next)
            x_next = or_float_tensors(x_next, x)

        # forward pass
        x_var = Variable(x_next, requires_grad=True)
        y_model = model(x_var)
        loss = loss_fct(y_model, y)

        # update worst loss and adversarial samples
        replace_flag = loss.data > loss_worst
        loss_worst[replace_flag] = loss.data[replace_flag]
        x_worst[replace_flag.unsqueeze(1).expand_as(x_worst)] = x_next[
            replace_flag.unsqueeze(1).expand_as(x_worst)
        ]

    if is_report_loss_diff:
        print(
            "Natural loss (%.4f) vs Adversarial loss (%.4f), Difference: (%.4f)"
            % (
                loss_natural.mean(),
                loss_worst.mean(),
                loss_worst.mean() - loss_natural.mean(),
            )
        )

    if x_worst.is_cuda:
        x_worst = x_worst.cpu()

    return x_worst


def grosse_k(x, y, model, loss_fct, k=25, is_report_loss_diff=False, use_sample=False):
    """
    Multi-step bit coordinate ascent using gradient of output, advancing in direction of maximal change
    :param use_sample:
    :param is_report_loss_diff:
    :param loss_fct:
    :param y:
    :param x: (tensor) feature vector
    :param model: nn model
    :param k: num of steps
    :return adversarial version of x (tensor)
    """

    if next(model.parameters()).is_cuda:
        x = x.cuda()
        y = y.cuda()

    y = Variable(y)

    # compute natural loss
    loss_natural = loss_fct(model(Variable(x)), y).data

    # keeping worst loss
    loss_worst = loss_natural.clone()
    x_worst = x.clone()

    output = None
    x_var = None
    x_next = None
    for t in range(k):
        if t == 0:
            # initialize starting point
            x_next = get_x0(x, use_sample)
        else:
            grad_vars = torch.autograd.grad(output[:, 0].mean(), x_var)
            grad_data = grad_vars[0].data

            # Only consider gradients for points of 0 value
            aug_grad = (1.0 - x_next) * grad_data
            val, _ = torch.topk(aug_grad, 1)
            x_update = (aug_grad >= val.expand_as(aug_grad)).float()

            # find the next sample with projection to the feasible set
            x_next = xor_float_tensors(x_update, x_next)
            x_next = or_float_tensors(x_next, x)

        x_var = Variable(x_next, requires_grad=True)
        output = model(x_var)

        loss = loss_fct(output, y)

        # update worst loss and adversarial samples
        replace_flag = loss.data > loss_worst
        loss_worst[replace_flag] = loss.data[replace_flag]
        x_worst[replace_flag.unsqueeze(1).expand_as(x_worst)] = x_next[
            replace_flag.unsqueeze(1).expand_as(x_worst)
        ]

    if is_report_loss_diff:
        print(
            "Natural loss (%.4f) vs Adversarial loss (%.4f), Difference: (%.4f)"
            % (
                loss_natural.mean(),
                loss_worst.mean(),
                loss_worst.mean() - loss_natural.mean(),
            )
        )

    if x_worst.is_cuda:
        x_worst = x_worst.cpu()

    return x_worst


def inner_maximizer(x, y, model, loss_fct, iterations=100, method="natural"):
    """
    A wrapper function for the above algorithim
    :param iterations:
    :param x:
    :param y:
    :param model:
    :param loss_fct:
    :param method: one of 'dfgsm_k', 'rfgsm_k', 'bga_k', 'bca_k', 'natural
    :return: adversarial examples
    """
    if method == "dfgsm_k":
        return dfgsm_k(x, y, model, loss_fct, k=iterations)
    elif method == "rfgsm_k":
        return rfgsm_k(x, y, model, loss_fct, k=iterations)
    elif method == "bga_k":
        return bga_k(x, y, model, loss_fct, k=iterations)
    elif method == "bca_k":
        return bca_k(x, y, model, loss_fct, k=iterations)
    elif method == "grosse":
        return grosse_k(x, y, model, loss_fct, k=iterations)
    elif method == "natural":
        return x
    elif method == "topk":
        return grams_topk(x, y, model, loss_fct, k=iterations)
    elif method == "topk_plus":
        return grams_topk_plus(x, y, model, loss_fct, k=iterations)
    elif method == "particle_swarm":
        return particle_swarm(x, y, model, loss_fct)
    else:
        raise Exception("No such inner maximizer algorithm")
