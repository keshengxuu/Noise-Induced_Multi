from __future__ import print_function
import pyspike as spk
from pyspike import SpikeTrain
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec



plt.rcParams['mathtext.sf'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'

#changing the xticks and  yticks fontsize for all sunplots
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)
plt.rc('font',size=7)


spike_trains = [spk.load_spike_trains_from_txt("spikestrain_JEE05/Sptraintimes_%d.txt"%fl,separator=',',
                                               edges=(0, 2000)) for fl in range(6)]
spike_trains15 = [spk.load_spike_trains_from_txt("spikestrain_JEE15/Sptraintimes_%d.txt"%fl,separator=',',
                                               edges=(0, 2000)) for fl in range(6)]


fig = plt.figure(1, figsize = (9,6))
plt.clf()
fl = 1

gs = GridSpec(nrows=2, ncols=6, left=0.05, right=0.90,top=0.965,bottom=0.575,wspace=0.15,hspace=0.35)
gs1 = GridSpec(nrows=2, ncols=6, left=0.05, right=0.90,top=0.50,bottom=0.05,wspace=0.15,hspace=0.35)

ax1= fig.add_subplot(gs[0, 0])
ax2= fig.add_subplot(gs[0, 1])
ax3= fig.add_subplot(gs[0, 2])
ax4= fig.add_subplot(gs[0, 3])
ax5= fig.add_subplot(gs[0, 4])
ax6= fig.add_subplot(gs[0, 5])

ax1b= fig.add_subplot(gs[1, 0])
ax2b= fig.add_subplot(gs[1, 1])
ax3b= fig.add_subplot(gs[1, 2])
ax4b= fig.add_subplot(gs[1, 3])
ax5b= fig.add_subplot(gs[1, 4])
ax6b= fig.add_subplot(gs[1, 5])


ax7 = fig.add_subplot(gs1[0, 0])
ax8 = fig.add_subplot(gs1[0, 1])
ax9= fig.add_subplot(gs1[0, 2])
ax10 = fig.add_subplot(gs1[0, 3])
ax11 = fig.add_subplot(gs1[0, 4])
ax12 = fig.add_subplot(gs1[0, 5])


ax7b= fig.add_subplot(gs1[1, 0])
ax8b = fig.add_subplot(gs1[1, 1])
ax9b = fig.add_subplot(gs1[1, 2])
ax10b = fig.add_subplot(gs1[1, 3])
ax11b = fig.add_subplot(gs1[1, 4])
ax12b = fig.add_subplot(gs1[1, 5])



axx= [ax1,ax2,ax3,ax4,ax5,ax6]
axxsp= [ax1b,ax2b,ax3b,ax4b,ax5b,ax6b]

axxx= [ax7,ax8,ax9,ax10,ax11,ax12]
axxxsp = [ax7b,ax8b,ax9b,ax10b,ax11b,ax12b]
binss =20
cmap00=plt.get_cmap('viridis_r')
for ax0, ax,i in zip(axx,axxsp,range(5)):
    spike_sync = spk.spike_sync_matrix(spike_trains[i], interval=(1000, 2000))
    ax0.imshow(spike_sync, vmin =0, vmax = 1.0, interpolation='none',cmap = cmap00)
    
    sp_sync= spike_sync[np.triu_indices(np.size(spike_sync,0))]
    hist, bins = np.histogram(sp_sync, bins=binss,range = (-0.1, 1.1))
    widths = np.diff(bins)
    ax.bar(bins[:-1],hist, widths)
    Var = np.var(sp_sync)
    ax.text(0.2, 0.8, r'$\mathsf{Var}=$'+'%0.2f'%(Var*1000),transform=ax.transAxes)

spike_sync = spk.spike_sync_matrix(spike_trains[5], interval=(1000, 2000))
im1 = ax6.imshow(spike_sync, vmin =0, vmax = 1.0, interpolation='none',cmap = cmap00)
#histgram 
sp_sync= spike_sync[np.triu_indices(np.size(spike_sync,0))]
hist, bins = np.histogram(sp_sync, bins=binss,range = (-0.1, 1.1))
widths = np.diff(bins)
ax6b.bar(bins[:-1],hist, widths)
Var = np.var(sp_sync)
ax6b.text(0.2, 0.8, r'$\mathsf{Var}=$'+'%0.2f'%(Var*1000),transform=ax6b.transAxes)


for ax0,ax,i in zip(axxx,axxxsp,range(6)):
    spike_sync = spk.spike_sync_matrix(spike_trains15[i], interval=(1000, 2000))
    ax0.imshow(spike_sync, vmin =0, vmax = 1.0, interpolation='none',cmap = cmap00)
    
    sp_sync= spike_sync[np.triu_indices(np.size(spike_sync,0))]
    hist, bins = np.histogram(sp_sync, bins=binss,range = (-0.1, 1.1))
    widths = np.diff(bins)
    ax.bar(bins[:-1],hist, widths)
    
    # Multistability var(spike_sync)
    Var = np.var(sp_sync)
    ax.text(0.2, 0.8, r'$\mathsf{Var}=$'+'%0.2f'%(Var*1000),transform=ax.transAxes)
    




for ax,ax0 in zip(axxsp,axxxsp):
   # ax.set_xticks([]) 
    ax.set_yticks([])
   # ax0.set_xticks([])  
    ax0.set_yticks([]) 
    
    
    
# colorbar
cax1 = fig.add_axes([0.92, 0.4, 0.015, 0.4]) #the parameter setting for colorbar position
cbar1=fig.colorbar(im1, cax=cax1)
#cbar1=fig.colorbar(im1, cax=cax1, orientation="horizontal")
#cbar1.set_label(r'$\mathsf{\chi}$',fontsize='x-large',labelpad=-13, x=-0.07, rotation=0,color='red')
cbar1.set_label(r'Spike-Sync',fontsize='medium',labelpad=2)
cbar1.set_ticks(np.linspace(0,1,3))
##change the appearance of ticks anf tick labbel
cbar1.ax.tick_params(labelsize='medium')
cax1.xaxis.set_ticks_position("top")
# #plt.legend()






axset = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12]
ayset = [ax1,ax7]

axticks = [ax2,ax3,ax4,ax5,ax6,ax8,ax9,ax10,ax11,ax12]
ayticks = [ax2,ax3,ax4,ax5,ax6,ax8,ax9,ax10,ax11,ax12]
lets=['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)']


for ax,let in zip(axset,lets):
    ax.set_xticks([0.0,500,1000])
    ax.set_xticklabels(['0','500','1000'])
    ax.set_xlabel('Spike Train',labelpad=0)
    ax.set_title('%s'%let)
    ax.tick_params(labelsize=7)
    
for ay in ayset:
    ay.set_yticks([0.0,500,1000])
    ay.set_yticklabels(['0','500','1000'])
    ay.set_ylabel('Spike Train',labelpad=-4)
    ay.tick_params(labelsize=7)
    
# for ax in axticks:
#     ax.set_xticks([])  

# ax1b.set_ylabel('Count numbers') 
# ax7b.set_ylabel('Count numbers') 
    
for ay in ayticks:
    ay.set_yticks([]) 
    
plt.show()



plt.figtext(0.012,0.956,r'$\mathsf{A}$',fontsize = 'x-large')
plt.figtext(0.012,0.48,r'$\mathsf{B}$',fontsize = 'x-large')

#plt.subplots_adjust(bottom=0.1,left=0.09,wspace = 0.2,hspace = 0.2,right=0.93, top=0.985)
plt.savefig("Figure5_SPIKE-Sync.png",dpi =300)
plt.savefig("Figure5_SPIKE-Sync.eps",dpi =300)
