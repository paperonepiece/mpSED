import math
import numpy as np
import matplotlib.pyplot as plt

def lorentzian(xarr,center,amplitude,hwhm):
    return amplitude/(1+((xarr-center)/hwhm)**2)

class plot:
    def __init__(self, data, params):
        self.sed_avg = data.sed_avg.copy()
        self.qpoints = data.qpoints.copy()
        self.thz = data.thz.copy()
        
        self._precision = 1e-5
        self._left_axis_split_num = 5
        self._cmap_color = 'afmhot'  # 'turbo', 'rainbow', 'jet', 'terrain', 'brg', 'inferno'
        self._cmap_interplot_method = 'hamming' # 'hanning', 'gaussian', 'catrom', 'spline36', 'lanczos'
        self._log_format_bar = True
        
        self.fre_cutoff = float(params.plot_cut)
        self.plot_name = params.out_prefix
        
    def plot_bands(self):
        thz = self.thz.copy()
        sed_avg = self.sed_avg.copy()
        qpoints = self.qpoints.copy()
        plot_name = self.plot_name + '_SED-band.png'
        
        if min(abs(thz - self.fre_cutoff)) > self._precision:
                print('\nCAUTION: User set Frequency cutoff {:.4f} Larger than upper limit {:.4f}, '
                        'Stay default as 0.00-{:.2f}'.format(self.fre_cutoff, thz.max(), thz.max()))
            
        cutoff_index = list(abs(thz - self.fre_cutoff)).index(min(abs(thz - self.fre_cutoff))) + 1 
                
        df_major = round(round(thz.max())/self._left_axis_split_num)
        if df_major == 0:
            # usually means upper limit is less than 2.5 THz
            df_major = 0.5
        freqs_values = np.arange(0, round(thz.max()) + self._precision, df_major) # axis_value-label
        mark_indexs = np.zeros(len(freqs_values))   # axis_split-mark     
        
        ## rescale label and marks on desired frequency domain
        if self.fre_cutoff != 0 and cutoff_index <= len(thz):
            print('\nPlot dispersion on cut Frequency domain: 0.00-%.2f THz'%(self.fre_cutoff), flush=True)
            thz = thz[:cutoff_index]
            sed_avg = sed_avg[:cutoff_index, :]
            # rescale axis label and mark
            df_major = round(round(thz.max())/self._left_axis_split_num)
            freqs_values = np.arange(0, round(thz.max()) + self._precision, df_major) # axis_label
            mark_indexs = np.arange(0, len(thz), len(thz) // (len(freqs_values)-1))
        else:
            print('\nPlot dispersion on original Frequency domain: 0.00-%.2f THz'%(thz.max()),flush=True)
            for i in range(len(freqs_values)):
                mark_indexs[i] = np.argwhere(thz <= freqs_values[i]).max()
        
        if self._log_format_bar:
            sed_avg = np.log10(sed_avg)  # 10**sed_avg = data.sed_avg
        
        ### creat a figure, set its size and draw basic IM type band-plot
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 6,forward=True)
        fig.tight_layout(pad=5)
        sed_min = np.trunc(sed_avg.min())
        sed_max = np.trunc(sed_avg.max())
        im = ax.imshow(sed_avg,
                        cmap=self._cmap_color,
                        interpolation=self._cmap_interplot_method,
                        aspect='auto',
                        origin='lower',
                        vmin=sed_min,
                        vmax=sed_max
        )
        
        fig.colorbar(im, 
                    ax=ax, 
                    label = r'log10($\Phi$($\omega)$)  (J*s)'
        )
        
        ### configure the plot
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)
            
        ## set x, y axis ticks and tick-labels
        ax.set_yticks(mark_indexs)
        ax.set_yticklabels(list(map(str, freqs_values)))
    
        xticks = [0, len(qpoints)-1]
        
        xlabels = ['']*len(xticks)
        for i in range(len(xticks)):
            xlabels[i] = '({:.1f},{:.1f},{:.1f})'.format(
                qpoints[xticks[i],0],
                qpoints[xticks[i],1],
                qpoints[xticks[i],2],
            )
        ax.set_xticks(xticks)    
        ax.set_xticklabels(xlabels)  
        ax.minorticks_on()
        
        ax.tick_params(which='both', axis='y', width=1, labelsize='large')
        ax.tick_params(which='both', axis='x', width=1, labelsize='large',
                        labelrotation=0.0, pad=5.0)
        ax.tick_params(which='major', length=5, color='k')
        ax.tick_params(which='minor', length=3, color='k')
        
        ## set x, y labels and fig. label
        ax.set_xlabel(r'$\mathbf{q}$',labelpad=5.0,fontweight='normal',fontsize='x-large')
        ax.set_ylabel(r'$\omega$ (THz)',labelpad=3.0,fontweight='normal',fontsize='x-large')
        fig.suptitle(r'$\Phi$($\mathbf{q}$,$\omega)$',y=0.95,fontsize='x-large')
        
        ### save figure
        plt.savefig(plot_name, format='png',dpi=300, bbox_inches='tight')
        
    def plot_q_slice(self, params):
        q_indexs = params.q_slice_index
        log_format_colorbar = True
    
        dfx_major = round(round(self.thz.max())/self._left_axis_split_num)
        freqs_values = np.arange(0, round(self.thz.max()) + self._precision, dfx_major)
        mark_indexs = np.zeros(len(freqs_values))
        for i in range(len(freqs_values)):
            mark_indexs[i] = np.argwhere(self.thz <= freqs_values[i]).max()
        
        if type(q_indexs) != list and type(q_indexs) != range:
            raise TypeError('\nERROR: Q_SLICE_INDEX must be integer or python range object\n')
        
        for q_index in q_indexs:
            ### creat a figure, set its size
            print('\nPlot SED on q-point {}'.format(q_index), flush=True)
            fig, ax = plt.subplots()
            fig.set_size_inches(6, 3, forward=True)
            fig.tight_layout(pad=3.5)
            
            # configure the plot
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(1.5)
    
            if log_format_colorbar:
                ax.semilogy(self.sed_avg[:, q_index],
                            ls='-',
                            lw=1,
                            color='orange',
                            marker='o',
                            ms=4.0,
                            mfc='none',
                            mec='royalblue',
                            mew=0.75)
                
                if params.plot_lorentz:
                    total = np.zeros(len(self.sed_avg[:, q_index]))
                    for i in range(len(params.popt[:,0])):
                        start = params.bounds[i,0]
                        end = params.bounds[i,1]
                        
                        if params.popt[i,2] == 0:
                            continue
                        
                        ax.semilogy(
                            lorentzian(
                                np.arange(len(self.sed_avg[:, q_index])),
                                params.popt[i,0],
                                params.popt[i,1],
                                params.popt[i,2],
                                ),
                                ls='-',lw=1,marker='o',mfc='r',mec='r',ms=1,mew=0,color='r')
                        
                        total = total + (
                            lorentzian(
                                np.arange(len(self.sed_avg[:, q_index])),
                                params.popt[i,0],
                                params.popt[i,1],
                                params.popt[i,2],
                                )
                            )
                        
                    ax.semilogy(
                        total,
                        ls='-',lw=0.5,marker='o',
                        mfc='b',mec='b',ms=0.5,mew=0,color='b')
    
            else:
                ax.plot(self.sed_avg[:, q_index],
                        ls='-',
                        lw=1,
                        color='k',
                        marker='o',
                        ms=2,
                        mfc='b',
                        mec='k',
                        mew=1)
                
                if params.plot_lorentz:
                    total = np.zeros(len(self.sed_avg[:, q_index]))
                    for i in range(len(params.popt[:,0])): 
                        start = params.bounds[i,0]
                        end = params.bounds[i,1]
                        if params.popt[i,2] == 0:
                            continue
                        
                        ax.plot(
                            lorentzian(
                                np.arange(len(self.sed_avg[:, q_index])),
                                params.popt[i,0],
                                params.popt[i,1],
                                params.popt[i,2],
                                ),
                                ls='-',lw=1,marker='o',mfc='r',mec='r',ms=1,mew=0,color='r'
                        )
                        
                        total = total + (
                            lorentzian(
                                np.arange(len(self.sed_avg[:, q_index])),
                                params.popt[i,0],params.popt[i,1],params.popt[i,2]
                                )
                        )
                    ax.plot(total,
                            ls='-',lw=0.5,marker='o',
                            mfc='b',mec='b',ms=0.5,mew=0,color='b')
             
            # xticks = np.arange(0, round(self.thz.max()) + self._precision, self._left_axis_split_num)
            # print(freqs_values)
            # print(round(self.thz.max()))
            xthicks = np.arange(0, len(self.thz)) / (len(self.thz) - 1) 
            xthicks_index = [0] * (self._left_axis_split_num+1)
            for i in range(1, self._left_axis_split_num+1):
                xthicks_index[i] = round(len(xthicks) * i / self._left_axis_split_num)
            xthicks_index[-1] -= 1
            # xthicks_label = [round(xthicks[xthicks_index[i]]*round(self.thz.max())) for i in range(6)]
            xthicks_label = [round(self.thz[xthicks_index[i]]) for i in range(6)]
            # print(xthicks_label)
            ax.set_xticks(xthicks_index)
            ax.set_xticklabels(xthicks_label)
            
            ax.minorticks_on()
            ax.tick_params(which='both', width=1)
            # ax.tick_params(which='both', width=1, labelsize='x-large')
            ax.tick_params(which='major', length=5)
            ax.tick_params(which='minor', length=3, color='k')
        #    plt.tick_params(axis='x',which='both',labelbottom=False)
            ax.set_ylabel(r'$\Phi$($\omega)$ (J*s)',
                          fontsize=10,
                          labelpad=8,
                          fontweight='normal',
                          rotation=90)
            
            # ax.set_ylabel(r'log($\Phi$($\omega)$)',
            #               fontsize=10,
            #               labelpad=8,
            #               fontweight='normal',
            #               rotation=90)
            
            ax.set_xlabel('Frequency (THz)',
                          labelpad=3.5,
                          fontweight='normal') # fontsize = 'large'
        #    ax.set_xlabel(r'$\omega$ (THz)',labelpad=3.0,fontweight='normal',fontsize='large')
            fig.suptitle(r'$\mathbf{q}$='+\
                        '({:.3f}, {:.3f}, {:.3f})'.format(
                            self.qpoints[q_index,0],
                            self.qpoints[q_index,1],
                            self.qpoints[q_index,2],
                            ),
                x=0.20,
                y=0.90,
            )
            
            plt.savefig(params.out_prefix+'_SED-q{:}.png'.format(q_index),
                        format='png',
                        dpi=300,
                        bbox_inches='tight'
            )
        
def plot_bands(data,params):
    sed_avg = data.sed_avg
    qpoints = data.qpoints
    thz = data.thz
    
    precision = 1e-5
    n_left_axis_split = 5
    
    color = 'afmhot'  # 'turbo', 'rainbow', 'jet', 'terrain', 'brg', 'inferno'
    interp = 'hamming' # 'hanning', 'gaussian', 'catrom', 'spline36', 'lanczos'
    log_format_colorbar = True
    
    ### define the frequency domain and left-axis labels
    fre_cutoff = float(params.plot_cut) # THz / 0 for not set
    if min(abs(thz - fre_cutoff)) > precision:
        print('\nCAUTION: User set Frequency cutoff {:.4f} Larger than upper limit {:.4f}, '
                'Stay default as 0.00-{:.2f}'.format(fre_cutoff, thz.max(), thz.max()))
    
    cutoff_index = list(abs(thz - fre_cutoff)).index(min(abs(thz - fre_cutoff))) + 1 
            
    df_major = round(round(thz.max())/n_left_axis_split)
    if df_major == 0:
        # usually means upper limit is less than 2.5 THz
        df_major = 0.5
    freqs_values = np.arange(0, round(thz.max()) + precision, df_major) # axis_value-label
    mark_indexs = np.zeros(len(freqs_values))   # axis_split-mark     
    
    ## rescale label and marks on desired frequency domain
    if fre_cutoff != 0 and cutoff_index <= len(thz):
        print('\nPlot dispersion on cut Frequency domain: 0.00-%.2f THz'%(fre_cutoff), flush=True)
        thz = thz[:cutoff_index]
        sed_avg = sed_avg[:cutoff_index, :]
        # rescale axis label and mark
        df_major = round(round(thz.max())/n_left_axis_split)
        freqs_values = np.arange(0, round(thz.max()) + precision, df_major) # axis_label
        mark_indexs = np.arange(0, len(thz), len(thz) // (len(freqs_values)-1))
    else:
        print('\nPlot dispersion on original Frequency domain: 0.00-%.2f THz'%(thz.max()),flush=True)
        for i in range(len(freqs_values)):
            mark_indexs[i] = np.argwhere(thz <= freqs_values[i]).max()
    
    if log_format_colorbar:
        sed_avg = np.log10(sed_avg)  # 10**sed_avg = data.sed_avg
    
    ### creat a figure, set its size and draw basic IM type band-plot
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 6,forward=True)
    fig.tight_layout(pad=5)
    sed_min = np.trunc(sed_avg.min())
    sed_max = np.trunc(sed_avg.max())
    im = ax.imshow(sed_avg,
                   cmap=color,
                   interpolation=interp,
                   aspect='auto',
                   origin='lower',
                   vmin=sed_min,
                   vmax=sed_max
    )
    
    fig.colorbar(im, 
                 ax=ax, 
                 label = r'log10($\Phi$($\omega)$)  (J*s)'
    )
    
    ### configure the plot
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        
    ## set x, y axis ticks and tick-labels
    ax.set_yticks(mark_indexs)
    ax.set_yticklabels(list(map(str, freqs_values)))

    xticks = [0, len(qpoints)-1]
    
    xlabels = ['']*len(xticks)
    for i in range(len(xticks)):
        xlabels[i] = '({:.1f},{:.1f},{:.1f})'.format(
            qpoints[xticks[i],0],
            qpoints[xticks[i],1],
            qpoints[xticks[i],2],
        )
    ax.set_xticks(xticks)    
    ax.set_xticklabels(xlabels)  
    ax.minorticks_on()
    
    ax.tick_params(which='both', axis='y', width=1, labelsize='large')
    ax.tick_params(which='both', axis='x', width=1, labelsize='large',
                   labelrotation=0.0, pad=5.0)
    ax.tick_params(which='major', length=5, color='k')
    ax.tick_params(which='minor', length=3, color='k')
    
    ## set x, y labels and fig. label
    ax.set_xlabel(r'$\mathbf{q}$',labelpad=5.0,fontweight='normal',fontsize='x-large')
    ax.set_ylabel(r'$\omega$ (THz)',labelpad=3.0,fontweight='normal',fontsize='x-large')
    fig.suptitle(r'$\Phi$($\mathbf{q}$,$\omega)$',y=0.95,fontsize='x-large')
    
    ### save figure
    plot_name = params.out_prefix+'.png'
    plt.savefig(plot_name, format='png',dpi=300,bbox_inches='tight')
    # plt.show()



def plot_slice(data,params):
    sed_avg = data.sed_avg
    qpoints = data.qpoints
    thz = data.thz
    q_indexs = params.q_slice_index

    log_format_colorbar = True

    df_major = 2
    freqs_values = np.arange(0, thz.max(), df_major)
    mark_indexs = np.zeros(len(freqs_values))
    for i in range(len(freqs_values)):
        mark_indexs[i] = np.argwhere(thz <= freqs_values[i]).max()
    
    if type(q_indexs) != list and type(q_indexs) != range:
        raise TypeError('\nERROR: Q_SLICE_INDEX must be integer or python range object\n')
    
    for q_index in q_indexs:
        ### creat a figure, set its size
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 4, forward=True)
        fig.tight_layout(pad=8)
        
        # configure the plot
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)

        if log_format_colorbar:
            ax.semilogy(sed_avg[:, q_index],ls='-',lw=1,color='k',
                    marker='o',ms=2,mfc='b',mec='k',mew=1)
            if params.plot_lorentz:
                total = np.zeros(len(sed_avg[:, q_index]))
                for i in range(len(params.popt[:,0])):
                    start = params.bounds[i,0]
                    end = params.bounds[i,1]
                    if params.popt[i,2] == 0:
                        continue
                    ax.semilogy(lorentzian(np.arange(len(sed_avg[:, q_index])),
                        params.popt[i,0],params.popt[i,1],params.popt[i,2]),
                        ls='-',lw=1,marker='o',mfc='r',mec='r',ms=1,mew=0,color='r')
                    total = total+(lorentzian(np.arange(len(sed_avg[:, q_index])),
                        params.popt[i,0],params.popt[i,1],params.popt[i,2]))
                ax.semilogy(total,ls='-',lw=0.5,marker='o',
                        mfc='b',mec='b',ms=0.5,mew=0,color='b')

        else:
            ax.plot(sed_avg[:, q_index],ls='-',lw=1,color='k',
                    marker='o',ms=2,mfc='b',mec='k',mew=1)
            if params.plot_lorentz:
                total = np.zeros(len(sed_avg[:, q_index]))
                for i in range(len(params.popt[:,0])): 
                    start = params.bounds[i,0]
                    end = params.bounds[i,1]
                    if params.popt[i,2] == 0:
                        continue
                    ax.plot(lorentzian(np.arange(len(sed_avg[:, q_index])),
                        params.popt[i,0],params.popt[i,1],params.popt[i,2]),
                        ls='-',lw=1,marker='o',mfc='r',mec='r',ms=1,mew=0,color='r')
                    total = total+(lorentzian(np.arange(len(sed_avg[:, q_index])),
                        params.popt[i,0],params.popt[i,1],params.popt[i,2]))
                ax.plot(total,ls='-',lw=0.5,marker='o',mfc='b',mec='b',
                        ms=0.5,mew=0,color='b')

        ax.minorticks_on()
        ax.tick_params(which='both', width=1, labelsize='x-large')
        ax.tick_params(which='major', length=5)
        ax.tick_params(which='minor', length=3, color='k')
    #    plt.tick_params(axis='x',which='both',labelbottom=False)
        ax.set_ylabel(r'log($\Phi$($\omega)$)',labelpad=35.0,fontweight='normal',
                fontsize='x-large',rotation='horizontal')
        ax.set_xlabel('Index',labelpad=3.0,fontweight='normal',fontsize='large')
    #    ax.set_xlabel(r'$\omega$ (THz)',labelpad=3.0,fontweight='normal',fontsize='large')
        fig.suptitle(r'$\mathbf{q}$='+\
                    '({:.3f}, {:.3f}, {:.3f})'.format(
                                            qpoints[q_index,0],
                                            qpoints[q_index,1],
                                            qpoints[q_index,2],
                        ),
            y=0.80,
            fontsize='x-large',
        )
        
        plt.savefig('q_slice_q{:}.png'.format(q_index),
                    format='png',
                    dpi=300,
                    bbox_inches='tight'
        )
        
        # plt.show()
    
