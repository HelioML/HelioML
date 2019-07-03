from irisreader.utils.date import from_Tformat
from irisreader.utils.date import to_epoch
import matplotlib.pyplot as plt
import numpy as np
import datetime

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML

from irisreader.data.mg2k_centroids import assign_mg2k_centroids, LAMBDA_MIN, LAMBDA_MAX
from irisreader.coalignment import find_closest_raster
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import NearestCentroid


def animate_mask(obs, obs_type, centroids, clr_dic):
    
    """
    Returns an animation of the observation with a centroid mask overlay 
    
    Parameters
    ----------
        obs : irisreader.observation
        obs_type : str
            allows us to set specific parameters for each observation
        centroids : float
            centroids found by k-means [n_centroids, lambda]
        clr_dic : {int:str}
            dictionary to color code each centroid 

    """

    sji = obs.sji[1]
    
    # extract only the most interesting part of an observation with the cut method
    if obs_type == 'qs':
        try: sji.cut( 0, 10 )
        except: pass
        gamma = .1 # control contrast

    if obs_type == 'f1':
        try: sji.cut( 1086, 1244 )
        except: pass
        gamma = .4
        
    if obs_type == 'f2':
        try: sji.cut( 0, 50 )
        except: pass
        gamma = .1

    raster = obs.raster("Mg II k")
    raster_inds = [find_closest_raster( raster, sji, i )[0] for i in range(sji.n_steps)]
    
    #------------ fucntion for labeling profiles with there closest centroids --------------
    
    def assign_mg2k_centroids( X, centroids ):

        centroid_ids = list( range( centroids.shape[0] ) )

        # check whether X comes in the correct dimensions
        if not X.shape[1] == centroids.shape[1]:
            raise ValueError( "Expecting X to have shape (_,{}). Please interpolate accordingly (More information with 'help(assign_mg2k_centroids)').".format( centroids.shape[1] ) )

        # create nearest centroid finder instance and fit it
        knc = NearestCentroid()
        knc.fit( X=centroids, y=centroid_ids )

        # predict nearest centroids for the supplied spectra
        # (making sure that X is normalized)
        assigned_mg2k_centroids = knc.predict( normalize(X) )

        # return vector of assigned centroids

        return assigned_mg2k_centroids 

    #---------------------------- label data ---------------------------

    labels = []

    for i in raster_inds:
        X = raster.get_interpolated_image_step( 
                    step = i, 
                    lambda_min = LAMBDA_MIN, 
                    lambda_max = LAMBDA_MAX, 
                    n_breaks = 216  
        )

        labels.append( assign_mg2k_centroids( X, centroids ) )

    #---------------------------- set up animation ---------------------------
    n = sji.shape[0]

    interval_ms = 100
    N = sji.shape[1]

    # initialize plot
    image = sji.get_image_step( 0 ).clip(min=0.01)**gamma
    sltxpos = sji.get_slit_pos( 0 )
    sltypos = np.linspace( 0, image.shape[0]-1, N, dtype=np.int )
    sji_flare_mask=labels[0][sltypos]
    clr_mask = [clr_dic.get(sji_flare_mask[i],'grey') for i in range(len(sji_flare_mask))]
    fig = plt.figure( figsize=(10,10) )
    im = plt.imshow( -image, cmap="Greys", origin='lower' )
    scat = plt.scatter( [sltxpos]*N, sltypos, c=clr_mask, marker='s', s=10, alpha=1)

    # do nothing in the initialization function
    def init():
        return im, scat

    # animation function
    def animate(i):
        sltxpos = sji.get_slit_pos( i )
        sji_flare_mask=labels[i][sltypos] # just some random shit
        xcenix = sji.headers[i]['XCENIX']
        ycenix = sji.headers[i]['YCENIX']
        date_obs = sji.headers[i]['DATE_OBS']
        im.axes.set_title( "Frame {}: {}".format( i, date_obs ) )
        im.set_data( -sji.get_image_step( i ).clip(min=0.01)**gamma )
        scat.set_offsets( np.array([ [sltxpos]*N, sltypos ] ).T )
        scat.set_array( sji_flare_mask )
        clr_mask = [clr_dic.get(sji_flare_mask[i],'grey') for i in range(len(sji_flare_mask))]
        scat.set_color( clr_mask )
        return im, scat

    # Call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=sji.n_steps, interval=interval_ms, blit=True)

    # Close the plot
    plt.close(anim._fig)

    # Show animation in notebook
    return HTML(anim.to_html5_video())


def preprocess_obs( obs, obs_type, lambda_min, lambda_max, n_bins, start_stop_inds=None, length_minutes=25, flare_margin_minutes=1, savedir="level_2A/", data_format="hdf5", verbosity_level=1 ):
    """
    verbosity level : int
        0: no output
        1: print shapes
        2: print flare diagnostics
    """

    raster = obs.raster("Mg II k")
    
    if verbosity_level == 1:
        print("\n---------------------------------------------------------------- Flare Events -------------------------------------------------------------------------------------------------------------------\n")
        # plot flare locations
        obs.goes.events.plot_flares()
        plt.show()
        
        # get closest flare
        flares = obs.goes.events.get_flares()
        mx_flares = obs.goes.events.get_flares( classes="MX" )
        
        if len( flares ) > 0:
            display(HTML(flares.to_html()))
            if len( mx_flares ) > 0:
                closest_flare = mx_flares.iloc[0]
                print( "Start time of closest M/X flare: {} (distance {} arcsec)".format( closest_flare['event_starttime'], np.round( closest_flare['dist_arcsec'], 2 ) ) )
            else:
                print( "No M/X flares within range." )
        else:
            print( "No flares within range." )       
        
        print("\n---------------------------------------------------------------- Proposed time cut -------------------------------------------------------------------------------------------------------------\n")
    
    if start_stop_inds is None:
        start, stop, delta_t = time_cut( obs, flare=(obs_type=="PF"), length_minutes=length_minutes, flare_margin_minutes=flare_margin_minutes )
    else:
        start = start_stop_inds[0]
        stop = start_stop_inds[1]
        delta_t = ( from_Tformat( raster.get_raster_pos_headers( raster_pos=raster.n_raster_pos-1 )[stop-1]['DATE_OBS'] ) - from_Tformat( raster.get_raster_pos_headers( raster_pos=0 )[start]['DATE_OBS'] ) ).seconds/60
    
    if verbosity_level >= 2:
        plot_goes_flux( obs, start, stop )
        
    if verbosity_level >= 1:
        print("index interval for raster position 0: {}-{} (totalling {} exposures, {} minutes)".format(start, stop, stop-start, np.round(delta_t, 1 ) ) )

    raster.cut( raster.get_global_raster_step( raster_pos=0, raster_step=start ), raster.get_global_raster_step( raster_pos=0, raster_step=stop ) )
    
    if verbosity_level >= 2:
        display( animate( obs ) )
        url = obs.get_hek_url(html=False)
        display( HTML( "<a href="+ url + ">link to HEK data</a>" ) )
    
    data = get_interpolated_raster_data( raster, lambda_min, lambda_max, n_bins )
        
    if verbosity_level >= 1:
        full_obs_timedelta = np.round( ( from_Tformat( raster.time_specific_headers[-1]['DATE_OBS'] ) - from_Tformat( raster.time_specific_headers[0]['DATE_OBS'] ) ).seconds/60, 1 )
        print("Raster is {} minutes long after cut (should be equal to {} minutes estimated above)".format( full_obs_timedelta, delta_t, 2 ) )
        print( "data Shape: {}, raster shape: {}, n_raster_pos: {}".format( data.shape, raster.shape, raster.n_raster_pos ) )
    
    try:
        sji = obs.sji("Mg II h/k 2796")
    except:
        sji = obs.sji("Si")
        
    return data, raster, sji
        

        
def extract_flare( obs, lambda_min, lambda_max, n_bins, length_minutes=25, flare_margin_minutes=1, savedir="level_2A/", data_format="hdf5", verbosity_level=1 ):
    
        raster = obs.raster("Mg II k")
        
        # get event info of closest flare
        mx_flares = obs.goes.events.get_flares( classes="MX" )
        closest_flare = mx_flares.iloc[0]
        
        # extract raster_pos=0 indices of flare start and stop
        start_time = from_Tformat( closest_flare['event_starttime'] )
        stop_time = from_Tformat( closest_flare['event_endtime'] )
        ts = np.array( raster.get_timestamps( raster_pos=0 ) )
        
        start = np.argmin( np.abs(ts-to_epoch(start_time) ) )
        stop = np.argmin( np.abs(ts-to_epoch(stop_time) ) )
            
        eff_start_time = from_Tformat( raster.get_raster_pos_headers( raster_pos=0 )[start]['DATE_OBS'] )
        eff_stop_time = from_Tformat( raster.get_raster_pos_headers( raster_pos=raster.n_raster_pos-1 )[stop]['DATE_OBS'] )

        # plot GOES curve
        if verbosity_level >= 2:
            plot_goes_flux( obs, start, stop )
        
        if verbosity_level >= 1:
            print( "Cutting raster from indices {}-{} --> {:.1f} minutes (of {:.1f} minutes total flare duration)".format( start, stop, (eff_stop_time-eff_start_time).seconds/60, (stop_time-start_time).seconds/60 ) )
        
        # cut the raster
        raster.cut( raster.get_global_raster_step( raster_pos=0, raster_step=start ), raster.get_global_raster_step( raster_pos=0, raster_step=stop+1 ) )
        
        # show animation
        if verbosity_level >= 2:
            display( animate( obs ) )
            url = obs.get_hek_url(html=False)
            display( HTML( "<a href="+ url + ">link to HEK data</a>" ) )

        # save the data
#         print( "Saving data.." )
        data = get_interpolated_raster_data( raster, lambda_min, lambda_max, n_bins )
#         save_data( data, raster.headers, savedir, "{}_{}".format( "FL", obs.full_obsid ) )
        
        if verbosity_level >= 1:
            full_obs_timedelta = np.round( ( from_Tformat( raster.time_specific_headers[-1]['DATE_OBS'] ) - from_Tformat( raster.time_specific_headers[0]['DATE_OBS'] ) ).seconds/60, 1 )
            print("Raster is {} minutes long after cut".format( full_obs_timedelta ) )
            print( "data Shape: {}, raster shape: {}, n_raster_pos: {}".format( data.shape, raster.shape, raster.n_raster_pos ) )
            
        try:
            sji = obs.sji("Mg II h/k 2796")
        except:
            sji = obs.sji("Si")
        
        return data, raster, sji
            
            
def plot_goes_flux( obs, i_start, i_stop ):

    raster = obs.raster("Mg II k")
    start_date = obs.start_date
    end_date = obs.end_date
    th = raster.get_raster_pos_headers( raster_pos=0 )
    th_n = raster.get_raster_pos_headers( raster_pos=raster.n_raster_pos-1 )
    start_cut_date = from_Tformat( th[i_start]['DATE_OBS'] )
    stop_cut_date = from_Tformat( th_n[i_stop]['DATE_OBS'] )
    
    # put time cut into green area
    ax = obs.goes.xrs.data.plot( y='B_FLUX', logy=True, label="GOES X-ray Flux", figsize=(24,5), lw=2 )
    ax.axvspan( start_cut_date, stop_cut_date, alpha=0.2, color='green' )
    ax.axvline( x=start_cut_date, color='green', linestyle='--', linewidth=3.0 )
    ax.axvline( x=stop_cut_date, color='green', linestyle='--', linewidth=3.0 )
    ax.set_xlim([start_date, end_date])
    plt.text( start_cut_date, 1e-1, i_start, fontsize=14, color="green", ha="center" )
    plt.text( stop_cut_date, 1e-1, i_stop+1, fontsize=14, color="green", ha="center" )

    ax.axhline( y=1e-4, color='black', linestyle='--', linewidth=1.0 )
    ax.axhline( y=1e-5, color='black', linestyle='--', linewidth=1.0 )
    ax.axhline( y=1e-6, color='black', linestyle='--', linewidth=1.0 )
    ax.axhline( y=1e-7, color='black', linestyle='--', linewidth=1.0 )
    ax.axhline( y=1e-8, color='black', linestyle='--', linewidth=1.0 )
    
    steps = obs.raster("Mg II k").get_raster_pos_steps( raster_pos=0 )
    if steps<=250:
        gridstep=1
    elif steps<=2500:
        gridstep=10
    else:
        gridstep=100
        
    for i in range( 0, len(th), gridstep ):
        ax.axvline( x=th[i]['DATE_OBS'], color='black', linestyle='--', linewidth=1.0 )
    
    ax.set_ylim([1e-9, 1e-2])
    ax.set_ylabel(r'Watts / m$^2$')
    ax.set_xlabel("Universal Time")
    
    ax2 = ax.twinx()
    ax2.set_yscale( 'log' )
    ax2.set_ylim( ax.get_ylim() )
    ax2.set_yticks([3e-8, 3e-7, 3e-6, 3e-5, 3e-4])
    ax2.set_yticklabels(['A', 'B', 'C', 'M', 'X'])
    ax2.minorticks_off()
    ax2.tick_params( right=False )
    
    ax3 = ax.twiny()
    ax3.set_xlim( ax.get_xlim() )
    xticks = [from_Tformat(th[i]['DATE_OBS']) for i in np.arange( 0, len(th), 10*gridstep )]
    ax3.set_xticks( xticks )
    ax3.set_xticklabels( np.arange( 0, len(th), 10*gridstep )  )
    ax3.set_xlabel("Exposure")
    plt.show()
    
def animate( obs ):
    ts = np.array( obs.sji[0].get_timestamps() )
    start_date = obs.raster("Mg II k").time_specific_headers[0]['DATE_OBS']
    stop_date = obs.raster("Mg II k").time_specific_headers[-1]['DATE_OBS']
    i_start_sji = np.argmin( np.abs( ts - to_epoch( from_Tformat( start_date ) ) ) )
    i_stop_sji = np.argmin( np.abs( ts - to_epoch( from_Tformat( stop_date ) ) ) )
    return obs.sji[0].animate( index_start=i_start_sji, index_stop=i_stop_sji, cutoff_percentile=99.0 )


def get_interpolated_raster_data( raster, lambda_min, lambda_max, n_bins ):
    """
    Returns data in the shape [n_raster_pos, n_steps_per_raster, n_y_pix, n_lambda_pix]
    
    Parameters
    ----------
        raster : irisreader.raster_cube
            already time cut raster cube object
        lambda_min : float
            minimum wavelength for interpolation
        lambda_max : float
            maximum wavelength for interpolation
        n_bins : int
            number of bins for interpolation
    """
    
    # empty array with the desired shape
    X = np.empty( shape=( raster.n_raster_pos, raster.shape[0], raster.shape[1], n_bins ) )

    for raster_pos in range( raster.n_raster_pos ):
        for step in range( raster.get_raster_pos_steps( raster_pos ) ):
            X[ raster_pos, step, :, :] = raster.get_interpolated_image_step( 
                step, raster_pos=raster_pos, lambda_min=lambda_min, lambda_max=lambda_max, n_breaks=n_bins
        )
    
    return X


def time_cut( obs, flare=False, length_minutes=25, flare_margin_minutes=1 ):
    
    raster = obs.raster("Mg II k")
    ts = np.array( raster.get_timestamps( raster_pos=0 ) )
    
    if flare:
        # find end of preflare in terms of raster_pos=0 raster that happens _before_ the flare (taken care of in find_preflare_end)
        stop = find_preflare_end( obs, margin_minutes=flare_margin_minutes )
        
        # find closest raster_pos=0 raster to suggested start time
        start_time = ts[stop] - length_minutes*60
        start = np.argmin( np.abs( ts - start_time ) )
        
    else:
        # set start to first raster sweep
        start = 0
        
        # find closest raster_pos=0 raster to start time at first raster sweep
        start_time = ts[0]
        stop = np.argmin( np.abs( ts - (start_time + length_minutes*60) ) )
    
    start_date = from_Tformat( raster.get_raster_pos_headers( raster_pos=0 )[start]['DATE_OBS'] )
    stop_date = from_Tformat( raster.get_raster_pos_headers( raster_pos=raster.n_raster_pos-1 )[stop-1]['DATE_OBS'] )
    delta_t = stop_date - start_date
    return [start, stop, np.round( delta_t.seconds/60, 1 )]


def find_sji_inds( obs ):
    ts = np.array( obs.sji[0].get_timestamps() )
    start_date = obs.raster("Mg II k").time_specific_headers[0]['DATE_OBS']
    stop_date = obs.raster("Mg II k").time_specific_headers[-1]['DATE_OBS']
    i_start_sji = np.argmin( np.abs( ts - to_epoch( from_Tformat( start_date ) ) ) )
    i_stop_sji = np.argmin( np.abs( ts - to_epoch( from_Tformat( stop_date ) ) ) )
    return i_start_sji, i_stop_sji