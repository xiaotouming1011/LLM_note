PPO

**近端策略优化（Proximal Policy Optimization, PPO）** 是OpenAI于2017年提出的一种强化学习算法，属于策略梯度（Policy Gradient）方法。

## 4+2理解法

**四个模型：策略模型、价值模型、奖励模型、参考模型**

策略模型：待优化的模型，参与参数更新

价值模型：计算当前动作和状态的期望回报，可由奖励模型或策略模型初始化而来，参与参数更新

奖励模型：计算当前动作的即时奖励，不参与参数更新

参考模型：由策略模型进行初始化，不参与参数更新，用于限制策略模型在优化时不偏离原始模型太远

**两个损失：策略损失、价值损失**

策略损失：用于优化策略模型

价值损失：用于优化价值模型

## 在搞清楚PPO之前，我们需要了解如下基本概念

**策略**

在RLHF中，策略即我们要优化的大模型，从策略进行采样的过程即大模型生成句子的过程。

**奖励**

**单步奖励**

根据当前状态、动作和下一个状态由奖励模型得到的即时奖励，评估当前动作的好坏。

**累计奖励**

一条完整轨迹的单步奖励累积之和。

**折扣奖励**

平衡即时奖励和长期奖励之间的关系，使得智能体在决策时不仅要考虑当前的奖励，还要考虑未来的潜在奖励。

**轨迹**

轨迹由一系列的状态、动作组成，代表一次完整的采样，即大模型生成一条完整的句子。

$$\tau = \left( s_{0}, a_{0}, s_{1}, a_{1}, \ldots, s_{T-1}, a_{T-1}\right)$$

**基于策略的强化学习的优化目标**

$$\begin{aligned}\arg\max_{\pi_{\theta}} J(\pi_{\theta}) &= \arg\max_{\pi_{\theta}}\mathbb{E}_{\tau \sim \pi_{\theta}} [R(\tau)] \\&= \sum_{\tau} R(\tau) P(\tau | \pi_{\theta}) \end{aligned}$$

优化目标即最大化每条轨迹的期望奖励。

计算梯度：

$$\begin{aligned}\nabla J(\pi_{\theta}) &= \sum_{\tau} R(\tau) \nabla P(\tau | \pi_{\theta}) \\&= \sum_{\tau} R(\tau) P(\tau | \pi_{\theta}) \frac{\nabla P(\tau | \pi_{\theta})}{P(\tau | \pi_{\theta})} \\&= \sum_{\tau} R(\tau) P(\tau | \pi_{\theta}) \nabla \log (P(\tau | \pi_{\theta})) \\&= \mathbb{E}_{\tau \sim \pi_{\theta}} [R(\tau) \nabla \log (P(\tau | \pi_{\theta}))]\end{aligned}$$

轨迹是从策略中采样得到，由系列状态和动作得到，所以

$$P(\tau | \pi_{\theta}) = \rho_{0}(s_{0}) \prod_{t=0}^{T-1} P(s_{t+1} | s_{t}, a_{t}) \pi_{\theta}(a_{t} | s_{t})$$

于是

$$\begin{aligned}\nabla \log (P(\tau | \pi_{\theta})) &= \nabla \left[ \log \rho_{0}(s_{0}) + \sum_{t=0}^{T-1} \log P(s_{t+1} | s_{t}, a_{t}) + \sum_{t=0}^{T-1} \log \pi_{\theta}(a_{t} | s_{t}) \right] \\ \end{aligned}$$

前两项和策略模型的参数$$\theta$$无关，可舍去，于是

$$\begin{aligned}\nabla \log (P(\tau | \pi_{\theta})) &= \sum_{t=0}^{T-1} \nabla \log \pi_{\theta}(a_{t} | s_{t}) \end{aligned}$$

最终得到

$$\begin{aligned}\nabla J(\pi_{\theta}) &= \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ R(\tau) \nabla \log (P(\tau | \pi_{\theta})) \right] \\&= \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ R(\tau) \sum_{t=0}^{T-1} \nabla \log \pi_{\theta}(a_t | s_t) \right]\end{aligned}$$

$$R(\tau)$$代表的是一整条轨迹的奖励，$$\pi_{\theta}(a_t | s_t)$$却是针对单步的，用整条轨迹的奖励去评估单步的价值，是否合适？

毕竟，单条轨迹的奖励高，不代表轨迹中每一步的奖励都高。所以，需要在$$R(\tau)$$上做文章，我们可以通过各种方法来代替$$R(\tau)$$。

$$\begin{aligned}\nabla J(\pi_{\theta}) &= \mathbb{E}_{\tau \sim \pi_{\theta}} \big[ \sum_{t=0}^{T-1} \Psi_{t} \nabla \log \pi_{\theta}(a_{t} | s_{t}) \big] \\ \end{aligned}$$

$$\Psi_{t}$$可以有如下不同的实现方式：

$$\begin{aligned} 1.\quad & \sum_{t=0}^{\infty} r_{t} &\text{轨迹的累积奖励} \\ 2.\quad & \sum_{t'=t}^{\infty} \gamma^{t'} r_{t'} &\text{轨迹的折扣奖励} \\ 3.\quad & \sum_{t'=t}^{\infty} \gamma^{t'} r_{t'} - b(s_{t}) &\text{引入基线} \\ 4.\quad & Q^{\pi}(s_{t}, a_{t}) &\text{动作价值函数} \\ 5.\quad & A^{\pi}(s_{t}, a_{t}) &\text{优势函数} \\ 6.\quad & r_{t} + \lambda V^{\pi}(s_{t+1}) - V^{\pi}(s_{t}) &\text{时序差分残差}\end{aligned}$$

## PPO是怎么演变而来的呢？

策略损失

累积折扣奖励的定义：

$$\begin{aligned} G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \ldots + \gamma^{T-t-1} r_{T-1}&=r_t + \gamma G_{t+1}\end{aligned}$$

动作价值函数：

$$\begin{aligned} Q_{\pi}(s_t, a_t) &= \mathbb{E}_{\pi} [G_t | s_t, a_t] \\&= \mathbb{E}_{\pi} \left[ \sum_{t=0}^{T-t} \gamma^t r_t | s_t, a_t \right] \\&= \mathbb{E}_{\pi} [r_t | s_t, a_t] + \mathbb{E}_{\pi} [\gamma V_{\pi}(s_{t+1}) | s_t, a_t] \\&= \sum_{s_{t+1} \in \mathcal{S}} P(s_{t+1} | s_t, a_t) R(s_t, a_t, s_{t+1}) + \gamma \sum_{s_{t+1} \in \mathcal{S}} P(s_{t+1} | s_t, a_t) V_{\pi}(s_{t+1}) \\&= \mathbb{E}_{s_{t+1} \sim P(\cdot | s_t, a_t)} [r + \gamma V_{\pi}(s_{t+1})]\end{aligned}$$

优势函数：

$$\begin{aligned} A_{\pi}(s_t, a_t) &= Q_{\pi}(s_t, a_t) - V_{\pi}(s_t) \\&= \mathbb{E}_{s_{t+1} \sim P(\cdot | s_t, a_t)} \left[ r_t + \gamma V_{\pi}(s_{t+1}) \right] - \mathbb{E}_{s_{t+1} \sim P(\cdot | s_t, a_t)} \left[ V_{\pi}(s_t) \right] \\&= \mathbb{E}_{s_{t+1} \sim P(\cdot | s_t, a_t)} \left[ r_t + \gamma V_{\pi}(s_{t+1}) - V_{\pi}(s_t) \right] \\&= \mathbb{E}_{s_{t+1} \sim P(\cdot | s_t, a_t)} \left[ TD\_error \right]\end{aligned}$$

如果$$V_{\pi}$$可以准确衡量出策略$${\pi}$$的价值，那么$$TD\_error$$就是优势函数的无偏估计，但是在PPO中，$$V_{\pi}$$是通过神经网络模型预测得到的，在模型没有收敛之前都是有偏的，这个时候就会引发高偏差问题。为了平衡偏差与方差，引入了GAE。

其公式如下：

$$\begin{aligned}\Psi_t &= \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l} \\&= \sum_{l=0}^{\infty} (\gamma \lambda)^l \left( r_{t+l} + \gamma V_{\pi}(s_{t+l+1}) - V_{\pi}(s_{t+l}) \right) \end{aligned}$$

$$\lambda$$趋于0时，$$\Psi_t = r_t + \gamma V_{\pi}(s_{t+1}) - V_{\pi}(s_{t})$$，高偏差

$$\lambda$$趋于1时，低偏差，高方差

$$\begin{aligned}\Psi_t &= \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l} \\&= \sum_{l=0}^{\infty} (\gamma \lambda)^l \left( r_{t+l} + \gamma V_{\pi}(s_{t+l+1}) - V_{\pi}(s_{t+l}) \right) \\&= r_t + \gamma V_{\pi}(s_{t+1}) - V_{\pi}(s_{t}) + \gamma r_{t+1} + \gamma ^2V_{\pi}(s_{t+2}) - \gamma V_{\pi}(s_{t+1}) + ... \\&= -V_{\pi}(s_t) + r_t + \lambda r_{t+1} + ... \end{aligned}$$

为什么此时是低偏差，高方差呢？

因为既然$$V_{\pi}$$预测不准，那么我们就减少对其的使用，更多的采用即时奖励$$r_t$$，此时偏差比较低，但是此时又引入了更多的随机变量，会带来高方差。

$$\lambda$$越小，方差越小，偏差越大。λ越大，方差越大，偏差越小。

通过超参数来平衡对$$V_{\pi}$$和即时奖励的使用，平衡方差和偏差，这就是GAE（广义优势估计）。

此外，

$$\begin{aligned}\nabla J(\pi_{\theta}) &= \mathbb{E}_{\tau \sim \pi_{\theta}} \big[ \sum_{t=0}^{T-1} \Psi_{t} \nabla \log \pi_{\theta}(a_{t} | s_{t}) \big] \\ \end{aligned}$$

根据上述公式可知，轨迹是从当前策略采样得到的，也就是使用当前策略生成的数据来计算梯度，优化该模型，是一种同策略模型，每次生成的数据只使用一次，这是非常浪费的，因为在PPO生成数据需要使用4个模型，如果每次数据只使用一次，成本是很高的。为了能够使用旧策略模型生成的数据来优化当前模型，我们需要引入重要性采样，这样优化目标就变成了如下形式：

$$\begin{aligned}\nabla J(\pi_{\theta}) &= \mathop{E_{t}}\limits_{\tau \sim \pi_{\theta^{old}}} \left[ \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta^{old}}(a_t | s_t)} \Psi_{t} \nabla \log \pi_{\theta}(a_t | s_t) \right]\\ &=\mathop{E_{t}}\limits_{\tau \sim \pi_{\theta^{old}}} \left[ \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta^{old}}(a_t | s_t)} \Psi_{t} \frac{\nabla \pi_{\theta}(a_t | s_t)}{\pi_{\theta}(a_t | s_t)}  \right]\\ \end{aligned}$$

那么

$$\begin{aligned}J(\pi_{\theta}) &= \mathop{E_{t}}\limits_{\tau \sim \pi_{\theta^{old}}} \left[ \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta^{old}}(a_t | s_t)} \Psi_{t} \right]\end{aligned}$$

此时，轨迹是从旧策略模型采样得到的，通过一个比例调节因子，可以用于优化当前模型。

$$\begin{aligned}\arg\max_{\pi_{\theta}} J(\pi_{\theta}) &= \arg\max_{\pi_{\theta}}\mathop{E_{t}}\limits_{\tau \sim \pi_{\theta^{old}}} \left[ \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta^{old}}(a_t | s_t)} \Psi_{t} \right]\end{aligned}$$

到了这里，还有个问题，那就是如果新旧策略的分布差异较大，会导致训练不稳定，怎么来改善这个问题呢，有两种方法：

1、clip裁剪

通过对$$\frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta^{old}}(a_t | s_t)}$$进行裁剪，将其限制在特定的范围内，避免其更新幅度过大，这就得到了策略损失的最终形式。

$$\begin{aligned}\arg\max_{\pi_{\theta}} J(\pi_{\theta}) &= \arg\max_{\pi_{\theta}} \mathop{E_{t}}\limits_{\tau \sim \pi_{\theta^{old}}} \left[ \min \left( \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta^{old}}(a_t | s_t)} \Psi_t, \text{clip}\left( \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta^{old}}(a_t | s_t)}, 1 - \epsilon, 1 + \epsilon \right) \Psi_t \right) \right]\end{aligned}$$

为什么还需要一个取最小化的操作呢？这样有助于在偏差和方差之间取得平衡。裁剪机制限制了策略比率的范围，减少了高方差的可能性，而最小化操作进一步确保了目标函数不会过度偏向于任何一方（不会过度奖励也不会过度惩罚），从而保持了估计的稳定性。

2、加入KL散度，这时候优化目标变成了如下形式：

$$\begin{aligned}\arg\max_{\pi_{\theta}} J(\pi_{\theta}) &= \arg\max_{\pi_{\theta}}\mathop{E_{t}}\limits_{\tau \sim \pi_{\theta^{old}}} \left[ \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta^{old}}(a_t | s_t)} \Psi_{t} - \beta \text{KL}(\pi_{\theta^{old}}(\cdot | s_t), \pi_{\theta}(\cdot | s_t)) \right]\end{aligned}$$

价值损失

价值损失采用的是MSE损失，最小化如下目标函数：

$$\mathcal{L}_{\text{critic}}(\phi) = \mathbb{E}_t \left[ \left( V_{\phi}(s_t) - R_t \right)^2 \right]$$

其中$$V_{\phi}(s_t)$$为价值模型预测出来的回报，$$R_t$$为实际得到的回报。

如果使用时序差分目标，则

$$R_t = r_t + \gamma V_\phi(s_{t+1})$$

如果使用GAE目标，则

$$R_t = \hat{A}_t^{\text{GAE}} + V'_\phi(s_t)$$

当然这里也可以加入裁剪策略。