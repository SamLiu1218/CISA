import numpy as np
import pandas as pd
import scipy
from skimage import morphology
import warnings

def my_formatwarning(msg, *args, **kwargs):
    return f"Warning: {msg}\n"
warnings.formatwarning = my_formatwarning

def CISA(img_dict, cell_df, cellmask, boundmask, neighbor_classes, expressing_markers, cell_list = None, target_classes = ['T'], n_bootstraps = 1000, e_width = 2, bound_width = 0, dropna=True, class_col = 'Class', cell_id_col = 'Cell', keep_cols = ['Intratumoral'], excluded = ['T']):
    """
    Calculates CISA scores between given center cell types and their neighboring cell types for given markers. The default inputs are for calculating synapses between T cells and neighboring cells. Make sure to check all input arguments if applied to other cell types.

    Output: a dataframe where each row is a contacting center-neighbor cell pair.

    Input (required):

        img_dict: {marker: 2d image}. Input single channel images as a dictionary.

        cell_df: dataframe of cell information where each row is a cell. The only required columns are cell_id and cell_type, with colnames customizable as optional input. 

        cellmask: 2D int ndarray. Image mask of cells. Pixel values should be integers that matches cell ID in cell_df.

        boundmask: 2D int ndarray. Image mask of cell boundaries. Pixel values should be integers that matches cell ID in cell_df.

        neighbor_classes: list of neighboring classes. Other classes are not included in the result dataframe, but overlaping areas are still considered as non-contacting areas.

        expressing_markers: list of markers that are expressed in the center cell types. Their background level will be calculated as the average level in other cell types. All other markers that are not listed but included in img_dict will be treated as expressed in the center cell types and their background level will be calculated as the average level in center cell types.

    Input (optional):
        cell_list: list of cell IDs of center cells to compute CISA. If None, all cells are considered.

        target_classes: list of center cell types. Default: ['T'].

        n_bootstrap: number of bootstraps for the null model. Default: 1000

        e_width: width of dilation to calculate overlapping. Default: 2 pixels.

        bound_width: width of cell boundary dilation width before calculating overlapping. 0 = use raw bound. Default: 0 pixel.

        dropna: Bool, if drops invalid synapses. Default: True

        class_col: the colname for the cell type column in cell_df. Default: 'Class'

        cell_id_col: the colname for the cell ID column in cell_df that matches the cell IDs in the masks. Default: 'Cell'

        keep_cols: list of colnames that should be kept in the output dataframe. Default: []
        
        excluded: list of cell between which and the center cells the contacted regions should be completely ignored in calculation. One example is to reduce the influence of signals from neighboring same type of cells. Default: ['T'].

    """

    cell_df = cell_df.copy()
    # < checks >
    # cell_id_col dtype
    if cell_df[cell_id_col].dtype != int:
        warnings.warn(f"Cell IDs in cell_df[{cell_df_col}] are not integers. Converted to integers.")
        cell_df[cell_id_col] = cell_df[cell_id_col].astype(int)

    # keeps_cols presence
    for m in keep_cols:
        if m in cell_df.columns:
            continue
        else:
            keep_cols = [c for c in keep_cols if c != m]
            warnings.warn(f"{m} is in keep_cols but not provided in cell_df. Removed from keep_cols.")

    # cells with missing cell types
    n_nancelltype = np.sum(cell_df[class_col].isna())
    
    if n_nancelltype > 0:
        warnings.warn(f"{n_nancelltype} cells have NaN cell types. NOT removed from analysis.")
        # cell_df = cell_df.loc(~cell_df[class_col].isna()).copy()
        
    # presence of asked cell types
    if np.sum(cell_df[class_col].isin(target_classes)) == 0:
        warnings.warn("None of target_classes is present in the data")


    if np.sum(cell_df[class_col].isin(neighbor_classes)) == 0:
        warnings.warn("None of neighbor_classes is present in the data")           
        
    
    # < /checks >

    markers = list(img_dict.keys())

    # < bgl >
    valarr = np.zeros(cellmask.max() + 1, dtype = int)
    valarr[cell_df.loc[cell_df[class_col].isin(target_classes), cell_id_col].values] = 1 # T
    valarr[cell_df.loc[~cell_df[class_col].isin(target_classes), cell_id_col].values] = -1 # non T 
    mask_T = valarr[cellmask] == 1
    mask_nonT = valarr[cellmask] == 1

    bgl_dict = {}
    for m in markers:
        if m in expressing_markers:
            bgl_dict[m] = np.sum(img_dict[m][mask_nonT])/np.sum(mask_nonT)
        else:
            bgl_dict[m] = np.sum(img_dict[m][mask_T])/np.sum(mask_T)
        if bgl_dict[m] <= 0:
            warnings.warn(f"{m} has a background level of bgl_dict[m].")
    # </ bgl >

    # < init >
    diam = morphology.diamond(e_width)
    headers = ['Cell','Class','Neighbor','Neighbor_Class','area_c','area_nc'] + list(keep_cols)
    headers = headers + [f"{m}_{h}" for h in ["c", "nc", "CISA", "miu","sigma", "z", "p"] for m in list(img_dict.keys()) ]
    syn_df = pd.DataFrame(columns=headers)
    # </ init >

    # < cell list >
    if cell_list is None:
        cell_list = cell_df.loc[cell_df[class_col].isin(target_classes), cell_id_col].values

    # </ cell list >

    # main
    n_missing = 0
    for cell_id in cell_list:
        # skip non existing cell id
        if np.sum(boundmask == cell_id) == 0:
            n_missing += 1
            continue

        # < neighbor list >
        bm, [ytop,ybot,xleft,xright] = maskcrop(boundmask, boundmask == cell_id, padding=10, return_range=True)
        
        bm_cur = bm == cell_id
        
        for w in range(bound_width):
            bm_cur = morphology.binary_dilation(bm_cur)
            
        neighbor_ls = np.unique(bm).astype(int)
        neighbor_ls = [i for i in neighbor_ls if i not in [0, cell_id]]   
        # </ neighobor list >

        # < exclusion >
        for neighbor_id in neighbor_ls:
            interactee_c = cell_df.loc[cell_df[cell_id_col] == neighbor_id, class_col].values[0]
            if interactee_c not in excluded:
                continue
            bm_nb = bm == neighbor_id 
            bm_nb_dil = morphology.binary_dilation(bm_nb, diam)
            mask_c = np.logical_and(bm_cur, bm_nb_dil)
            bm_cur = bm_cur^mask_c
        # </ exclusion >

        # < contacting & non contacting masks >
        dict_nc = {c:bm_cur for c in neighbor_classes}
        dict_c = {}
        
        # find mask_nc for each class
        for neighbor_id in neighbor_ls:
            interactee_c = cell_df.loc[cell_df[cell_id_col] == neighbor_id, class_col].values[0]
            if interactee_c not in neighbor_classes:
                continue

            bm_nb = bm == neighbor_id 

            bm_nb_dil = morphology.binary_dilation(bm_nb, diam)
            
            mask_c = np.logical_and(bm_cur, bm_nb_dil)
            
            # if not contacted at all
            if np.sum(mask_c) == 0:
                continue
                
            # if contacted by a tiny area, dilate by one additional pixel
            if np.sum(mask_c) < 3:
                bm_nb_dil = morphology.binary_dilation(bm_nb_dil, morphology.diamond(1))
                mask_c = np.logical_and(bm_cur, bm_nb_dil)
                
            mask_nc = np.logical_and(dict_nc[interactee_c], np.logical_not(bm_nb_dil))
            dict_nc[interactee_c] = mask_nc
            dict_c[neighbor_id] = mask_c
        # </ contacting & non contacting masks >
        
        # CISA and null model
        for neighbor_id in dict_c.keys():
            interactee_c = cell_df.loc[cell_df[cell_id_col] == neighbor_id, class_col].values[0]
            if interactee_c not in neighbor_classes:
                continue
            mask_nc = dict_nc[interactee_c] 
            mask_c = dict_c[neighbor_id] 
            
            area_c = np.sum(mask_c)
            area_nc = np.sum(mask_nc)
                
            if area_nc == 0:
                continue
            if area_c == 0:
                continue
                
            dict_new = {'Cell':cell_id, 
                        'Class':cell_df.loc[cell_df[cell_id_col]==cell_id, class_col].values[0], 
                        'Neighbor':neighbor_id, 
                        'Neighbor_Class':interactee_c}
            for c in keep_cols:
                if c in cell_df.columns:
                    dict_new[c] = cell_df.loc[cell_df[cell_id_col] == cell_id, c].values[0]

            dict_new['area_c'] = area_c
            dict_new['area_nc'] = area_nc
            
            null_df = calculate_CISA_null_multi(
                {m:img_dict[m][ytop:ybot, xleft:xright] for m in img_dict.keys()}, 
                bm, area_c, bgl_dict, n_bootstraps=n_bootstraps)
            for m in img_dict.keys():
                img = img_dict[m]
                bgl = bgl_dict[m]

                img_crop = img[ytop:ybot, xleft:xright]

                dict_new[f'{m}_c'] = np.sum(img_crop[mask_c])/area_c if area_c > 0 else bgl
                dict_new[f'{m}_nc'] = np.sum(img_crop[mask_nc])/area_nc if area_nc > 0 else bgl

                dict_new[f'{m}_CISA'] = calculate_CISA(img_crop, mask_c, mask_nc, bgl)


                dict_new[f'{m}_miu'] = null_df.at[m, 'miu']
                
                dict_new[f'{m}_sigma'] = null_df.at[m, 'sigma']

                if dict_new[f'{m}_sigma'] > 0:
                    dict_new[f'{m}_z'] = (dict_new[f'{m}_CISA'] - dict_new[f'{m}_miu'])/dict_new[f'{m}_sigma']
                else: 
                    dict_new[f'{m}_z'] = 0
                
                dict_new[f'{m}_p'] = scipy.stats.norm.sf(abs(dict_new[f'{m}_z']))

            syn_df = pd.concat([syn_df, pd.DataFrame(dict_new, index = [0])], ignore_index=1)

    if n_missing > 0:
        warnings.warning(f"{n_missing} cells do not have matching cell masks.")
    if dropna:
        return syn_df.dropna()
    else:
        return syn_df
    

def M2E(M, nonzero=True):
    '''
    Convert an adjacency matrix to an edge list.
    '''
    df = pd.DataFrame(M)
    df = df.stack().reset_index()
    if nonzero:
        return df.loc[df.iloc[:,2]>0, :].values[:,:2]
    else:
        return df.values[:,:2]

def maskcrop(im, mask=None, padding=None, return_range=False):
    mask = mask.copy()
    im_size = im.shape
    if type(mask)==type(None):
        mask = (im>0)
    mask = mask*1
    
    xleft = np.argwhere(np.any(mask, axis=0))[0][0]
    xright = np.argwhere(np.any(mask, axis=0))[-1][0]

    ytop = np.argwhere(np.any(mask, axis=1))[0][0]
    ybot = np.argwhere(np.any(mask, axis=1))[-1][0]

    if padding == None:
        padding = int(0.05*max(xright-xleft, ybot-ytop))
    xleft = int(max(xleft - padding, 0))
    xright = int(min(xright + padding, im_size[1]))
    ytop = int(max(ytop - padding, 0))
    ybot = int(min(ybot + padding, im_size[0]))
    if len(im.shape) == 3:
        im_new = im[ytop:ybot,xleft:xright,:]
    else:
        im_new = im[ytop:ybot,xleft:xright]
    if return_range:
        return im_new, [ytop,ybot,xleft,xright]
    else:
        return im_new
    
    
def calculate_CISA_null_multi(img_dict, bm, area_c, bgl_dict, n_bootstraps=1000):
    """
    returns a df in the form of [m, ['miu','sigma']]
    """
    df = pd.DataFrame(columns = ['sigma','miu'], index = list(img_dict.keys()))
    coord_list = M2E(bm)
    grow_seed = area_c
    
    box = [ ( x, y ) for x in [-1, 0, 1] for y in [-1, 0, 1] ]
    box.remove((0, 0))
    coord_set = set()
    for y, x in coord_list:
        coord_set.add((y, x))
    seed_idx = np.random.choice(np.arange(len(coord_list)), n_bootstraps)
    val_dict = {m:list() for m in img_dict.keys()}
    for fi, i in enumerate(seed_idx):
        s_y, s_x = coord_list[i]
        seed_coord = (s_y, s_x)
        s_coord = set()
        s_coord.add(seed_coord)
        next_seed = set()
        next_seed.add(seed_coord)
        while len(s_coord) <= grow_seed: # grow random pixels until reaching 15
            grow_set = set()
            for ny, nx in next_seed: # grow a 3*3 chunk about current pixel
                for dy, dx in box:
                    grow_set.add(( ny + dy, nx + dx ))
                    
            if len(grow_set) == 0:
                
                break
            grow_set &= coord_set
            next_seed = grow_set - s_coord
            s_coord |= grow_set
            
        mask_c = np.zeros_like(bm)
        for x,y in s_coord:
            mask_c[x,y] = 1
        mask_c = mask_c > 0
        mask_nc = np.logical_and(bm, np.logical_not(mask_c))
        for m in img_dict.keys():
            val_dict[m].append(calculate_CISA(img_dict[m], mask_c, mask_nc, bgl_dict[m]))
            
    for m in list(img_dict.keys()):
        try:
            df.at[m, 'miu'] = np.mean(val_dict[m])
            df.at[m, 'sigma'] = np.std(val_dict[m])
        except:
            pass
    return df
    

def calculate_CISA(img, mask_c, mask_nc, bgl):
    area_c = np.sum(mask_c)
    area_nc = np.sum(mask_nc)
    sum_c = np.sum(img[mask_c])
    sum_nc = np.sum(img[mask_nc])
    
    if area_c == 0 or area_nc == 0:
        return np.nan
    
    if sum_nc == 0:
        val_nc = bgl
    else:
        val_nc = sum_nc/area_nc
        
    if sum_c == 0:
        val_c = np.minimum(bgl, val_nc)
    else:
        val_c = sum_c/area_c

    return np.log2(np.array(val_c/val_nc, dtype='float32'))
