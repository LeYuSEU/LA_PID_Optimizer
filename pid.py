import warnings

warnings.filterwarnings('ignore')

import torch
from torch.optim.optimizer import Optimizer, required
import torch.nn as nn


class PIDOptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False, I=5., D=10.):
        # 这些变量将自动保存到 self.param_groups 中
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov, I=I, D=D)

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super(PIDOptimizer, self).__init__(params, defaults)

    def InitWeightsWith_Lr(self, model):
        self.weight_layer_lr = nn.ParameterDict()
        self.weight_layer = dict()

        for idx, (key, param) in enumerate(model.named_parameters()):
            # 每一层设置不同的学习率, 甚至可以设为每一层的学习率 可以自己学习
            self.weight_layer_lr[key.replace(".", "-")] = nn.Parameter(data=torch.tensor(self.param_groups[0]['lr'] * torch.exp(torch.tensor(-idx))), requires_grad=False)
            # 记录模型的每一层名称
            self.weight_layer[key.replace(".", "-")] = self.param_groups[0]['params'][idx]

    def __setstate__(self, state):
        super(PIDOptimizer, self).__setstate__(state)

        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None, gen_I=None, gen_D=None):
        """
            Performs a single optimization step.
            Arguments:
                closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None

        if closure is not None:
            loss = closure()

        # error = (self.weight_layer['fc1-weight'] - self.param_groups[0]['params'][0]).sum() + (self.weight_layer['fc1-bias'] - self.param_groups[0]['params'][1]).sum() +\
        #         (self.weight_layer['fc2-weight'] - self.param_groups[0]['params'][2]).sum() + (self.weight_layer['fc2-bias'] - self.param_groups[0]['params'][3]).sum()
        # print('*' * 50, f'error = {error}', '*' * 50)

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            I = group['I']
            D = group['D']

            for key, p in self.weight_layer.items():
            # for p in group['params']:  # p 是各个层的参数 即theta
                if p.grad is None:
                    continue

                p_grad = p.grad.data
                if weight_decay != 0:  # p_grad 是梯度，p是可学习参数
                    p_grad.add_(weight_decay, p.data)  # 正则化, p_grad = p_grad + weight_decay * p

                if momentum != 0:
                    param_state = self.state[p]

                    # ************************* I_buf 就是动量Vt ***************************************************
                    if 'I_buffer' not in param_state:  # 第一次先初始化 I_buffer
                        I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                        I_buf.mul_(momentum).add_(p_grad)  # 第一次直接赋值等于梯度，p_grad
                    else:
                        I_buf = param_state['I_buffer']
                        I_buf.mul_(momentum).add_(1 - dampening, p_grad)  # I_buf * momentum + (1 - dampening) * p_grad

                    # ************************* D_buf 是新加入的 ***************************************************
                    if 'grad_buffer' not in param_state:  # 第一次先初始化
                        g_buf = param_state['grad_buffer'] = torch.zeros_like(p.data)
                        g_buf = p_grad

                        D_buf = param_state['D_buffer'] = torch.zeros_like(p.data)
                        D_buf.mul_(momentum).add_(p_grad - g_buf)  # D_buf[0] * momentum + (p_grad - g_buf) = p_grad - g_buf = 0
                    else:
                        D_buf = param_state['D_buffer']
                        g_buf = param_state['grad_buffer']
                        D_buf.mul_(momentum).add_(1 - momentum, p_grad - g_buf)  # D_buf * momentum + (1 - momentum) * (p_grad - g_buf)
                        self.state[p]['grad_buffer'] = p_grad.clone()            # 对于下次来说，这次的梯度就是上一次的梯度(历史梯度)

                    # p_grad = p_grad + I * I_buf + D * D_buf
                    # p_grad = p_grad.add_(I, I_buf).add_(D, D_buf)     # PID系数手动设置
                    p_grad = p_grad + gen_I[key] * I_buf + gen_D[key] * D_buf  # 每层的PID系数自动生成

                # p.data.add_(-group['lr'], p_grad)  # 使用统一的学习率参数更新， -lr * p_grad
                p.data.add_(-self.weight_layer_lr[key], p_grad)  # 参数更新，每一层的学习率不一样
                # p.data = p.data - self.weight_layer_lr[key] * p_grad  # .add_(-self.weight_layer_lr[key], p_grad)
            # break
        # --------------------------------------------------------------------------------------------------------------------
        # for p in group['params']:  # p 是各个层的参数 即theta
        # if p.grad is None:
        #     continue

        # p_grad = p.grad.data
        # if weight_decay != 0:  # p_grad 是梯度，p是可学习参数
        #     p_grad.add_(weight_decay, p.data)  # 正则化, p_grad = p_grad + weight_decay * p
        #
        # if momentum != 0:
        #     param_state = self.state[p]
        # tot_lr = [self.weight_layer_lr[key] for key, _ in self.weight_layer.items()]
        # print(tot_lr)

        return loss
