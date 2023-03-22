from configuration.constants import n_folds, transforms_show
from configuration.dataframes import df_seg
from run.segmentation import run as seg_run
from show.segmentation import show as seg_show


## Segmentació
# Visualització
print('Saving segmentation images')
seg_show(df=df_seg, transform=transforms_show)
print('Images saved successfully')

# Entrenament # ToDO Problemes treballant amb varios processos i assignacióo de memoria amb CUDA.
for i in range(n_folds):
    print('Running fold', i)
    seg_run(fold=i)
    print('Finished fold', i)


