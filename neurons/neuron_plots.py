#Plots and animation for single_neuron.py (no machine learning in this file)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

POSITIVE,NEGATIVE='#2a78d6','#e34948'  # colors for the sign of a weight
LOSS_COLOR,START_COLOR='#2a78d6','#eb6834'  # colors for the loss and its starting point

def draw_network(ax,x_inp,state,title):
    ax.clear()
    ax.set(xlim=(-0.5,4.6),ylim=(-2.5,2.5),aspect='equal',title=title)
    ax.axis('off')

    def node(x,y,text,color='white',radius=0.33):
        ax.add_patch(plt.Circle((x,y),radius,facecolor=color,edgecolor='#52514e',zorder=2))
        ax.text(x,y,text,ha='center',va='center',zorder=3)

    def connection(start,end,value,name):
        color=POSITIVE if value>=0 else NEGATIVE
        ax.plot(*zip(start,end),color=color,lw=min(0.8+10*abs(value),9),zorder=1)
        x,y=0.6*np.array(start)+0.4*np.array(end)
        ax.text(x,y,f'{name}={value:+.2f}',ha='center',va='center',fontsize=8,zorder=4,
                bbox=dict(facecolor='white',edgecolor='none',pad=1))

    center=(2.5,0)
    heights=np.linspace(1.8,-1.8,len(x_inp))

    for i,(x,w,y) in enumerate(zip(x_inp,state['weights'],heights),1):
        connection((0,y),center,w,f'w{i}')
        node(0,y,f'{x:g}')

    connection((2.5,2),center,state['bias'],'b')
    node(2.5,2,'1')

    ax.annotate('',xy=(3.77,0),xytext=center,arrowprops=dict(arrowstyle='->'),zorder=1)
    node(*center,f"Σ → σ\ns={state['s']:.2f}",color='#f0efec',radius=0.6)
    # the output node gets darker as the output approaches 1
    node(4.1,0,'',color=plt.cm.Blues(0.6*state['output']))
    ax.text(4.1,0,f"{state['output']:.2f}",ha='center',va='center',zorder=3)

    ax.text(0,-2.4,'inputs',ha='center')
    ax.text(4.1,-0.65,'output',ha='center')

def plot_network(x_inp,history):
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),constrained_layout=True)
    fig.suptitle('The neuron (blue = positive, red = negative, thicker = larger)')

    draw_network(axes[0],x_inp,history[0],'Before training')
    draw_network(axes[1],x_inp,history[-1],f'After {len(history)-1} steps')

    return fig

def plot_loss(target,history,loss_fn,gradient_fn):
    fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    fig.suptitle(f'Sigmoid loss for target={target} (dots = training steps)')

    visited=np.array([state['s'] for state in history])
    limit=max(8,np.abs(visited).max()+1)
    s_vals=np.linspace(-limit,limit,400)

    for ax,fn in zip(axes,[loss_fn,gradient_fn]):
        curve,dots=fn(s_vals,target),fn(visited,target)
        ax.plot(s_vals,curve,color=LOSS_COLOR,lw=2)
        ax.plot(visited,dots,'o',color=LOSS_COLOR,ms=6,mec='white',mew=0.5)
        ax.plot(visited[0],dots[0],'D',color=START_COLOR,ms=9,label='start')
        ax.legend()

    axes[2].plot([state['loss'] for state in history],color=LOSS_COLOR,lw=2)

    axes[0].set(title='Loss',xlabel='s = w·x + b',ylabel='L')
    axes[1].set(title='Gradient (flat at both ends)',xlabel='s = w·x + b',ylabel='dL/ds')
    axes[2].set(title='Loss during training',xlabel='step',ylabel='L')

    for ax in axes:
        ax.grid(alpha=0.25)

    return fig

def animate_training(x_inp,history,interval=180):
    fig=plt.figure(figsize=(12,6),constrained_layout=True)
    grid=fig.add_gridspec(2,2,width_ratios=[1.3,1])
    ax_net=fig.add_subplot(grid[:,0])
    ax_bar=fig.add_subplot(grid[0,1])
    ax_loss=fig.add_subplot(grid[1,1])

    params=np.array([list(state['weights'])+[state['bias']] for state in history])
    losses=[state['loss'] for state in history]
    names=[f'w{i}' for i in range(1,len(x_inp)+1)]+['b']
    low,high=min(params.min(),0),max(params.max(),0)
    margin=0.15*(high-low)

    bars=ax_bar.bar(names,params[0],width=0.5)
    ax_bar.plot(names,params[0],'_',color='#52514e',ms=28,zorder=3,label='starting value')
    ax_bar.axhline(0,color='#52514e',lw=0.8)
    ax_bar.set_title('Weights and bias',loc='left')
    ax_bar.set_ylim(low-margin,high+margin)
    ax_bar.legend(loc='lower right',bbox_to_anchor=(1,1),frameon=False)  # above the bars

    ax_loss.plot(losses,color='#e1e0d9',lw=2)
    line,=ax_loss.plot([],[],color=LOSS_COLOR,lw=2)
    dot,=ax_loss.plot([],[],'o',color=LOSS_COLOR,ms=8)
    ax_loss.set(title='Sigmoid loss',xlabel='step',ylabel='L')

    def update(step):
        title=f'Step {step}/{len(history)-1}    loss={losses[step]:.3f}'
        draw_network(ax_net,x_inp,history[step],title)

        for bar,value in zip(bars,params[step]):
            bar.set_height(value)
            bar.set_color(POSITIVE if value>=0 else NEGATIVE)

        line.set_data(range(step+1),losses[:step+1])
        dot.set_data([step],[losses[step]])

    update(0)
    anim=FuncAnimation(fig,update,frames=len(history),init_func=lambda: update(0),
                       interval=interval,repeat_delay=1500)
    return fig,anim

def show_plots(x_inp,target,history,loss_fn,gradient_fn):
    plot_network(x_inp,history)
    plot_loss(target,history,loss_fn,gradient_fn)
    fig,anim=animate_training(x_inp,history)  # keep a reference, otherwise it stops
    plt.show()
