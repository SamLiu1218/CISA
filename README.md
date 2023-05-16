# CISA
 Computational Immune Synapse Analysis.

## Example

See example notebook using core A5 from the published melanoma IMC dataset by Hoch et al., 2022

Hoch, Tobias, et al. "Multiplexed imaging mass cytometry of the chemokine milieus in melanoma characterizes features of the response to immunotherapy." Science immunology 7.70 (2022): eabk1692.

## Dependency

The core functions use only ```numpy```, ```pandas```, ```scipy``` and ```skimage```.

This package takes pixelated masks for cell segmentation and cell boundaries. However, it is recommanded to use `shapely` and `rasterio` to manipulate vectorized cell masks (not demonstrated).

## Usage
### Input (required):

 - ```img_dict```: {marker: 2d image}. Input single channel images as a dictionary.

 - `cell_df`: dataframe of cell information where each row is a cell. The only required columns are cell_id and cell_type, with colnames customizable as optional input.

 - `cellmask`: 2D int ndarray. Image mask of cells. Pixel values should be **integers** that matches the cell IDs in `cell_df`.

 - `boundmask`: 2D int ndarray. Image mask of cell boundaries. Pixel values should be integers that matches cell ID in `cell_df`.

 - `neighbor_classes`: list of neighboring classes. Other classes are not included in the result dataframe, but overlaping 

 - `expressing_markers`: list of markers that are expressed in the center cell types. Their background level will be calculated as the average level in other cell types. All other markers that are not listed but included in img_dict will be treated as expressed in the center cell types and their background level will be calculated as the average level in center cell types.

### Input (optional):
 - `cell_list`: list of cell IDs of center cells to compute CISA. If None, all cells are considered.

 - `target_classes`: list of center cell types. Default: `['T']`.

 - `n_bootstrap`: number of bootstraps for the null model. Default: 1000

- `e_width`: width of dilation to calculate overlapping. Default: 2 pixels.

 - `bound_width`: width of cell boundary dilation width before calculating overlapping. 0 = use raw bound. Default: 0 pixel.

 - `dropna`: Bool, if drops invalid synapses. Default: True

 - `class_col`: the colname for the cell type column in `cell_df`. Default: 'Class'

 - `cell_id_col`: the colname for the cell ID column in `cell_df`. Default: 'Cell'

 - `keep_cols`: list of colnames that should be kept in the output dataframe. Default: `[]`

 - `excluded`: list of cell between which and the center cells the contacted regions should be completely ignored in calculation. One example is to reduce the influence of signals from neighboring same type of cells. Default: `['T']`.

    
## Output
Output is a dataframe where each row corresponds to a center-neighbor cell pair.

For each marker ```m``` in the input, there will be several columns:

```{m}_CISA``` is the calculated synapse strength between the two cells.

```{m}_c``` and ```{m}_nc```: Sum of pixel intensity in the contacting (```c```) and non-contacting (```nc```) boundary areas.

```{m}_miu``` and ```{m}_sigma```: Mean and std of the synapses under the null model where a random connected pixls along the cell boundary are selected as the contacting area.

```{m}_z``` and ```{m}_p```: The z-score and p-value for the observed synapse strength under the null model.

## Note

Image preprocessing, such as de-noising, signal transformation and normalization, is determined by the user. 

A large bootstrap number may significantly increase the running time. In most cases, the CISA score (as in ```{m}_CISA``` column) is sufficient to characterize the behaviour.