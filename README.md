# A Generative Adversarial Net Assisted Method for User Intention Recognition on Imbalanced Dataset
We will add more descriptions after our paper "A Generative Adversarial Net Assisted Method for User Intention Recognition on Imbalanced Dataset" been published.

this work is published in 2022 IEEE International Conference on Knowledge Graph (ICKG), you may site this by: 

[1]. Z. Li, Z. Jin, R. Wang, J. Ji and H. Liu, "A Generative Adversarial Net Assisted Method for User Intention Recognition on Imbalanced Dataset," 2022 IEEE International Conference on Knowledge Graph (ICKG), Orlando, FL, USA, 2022, pp. 157-163, doi: 10.1109/ICKG55886.2022.00027.

## Dataset
1. the dataset used in our paper is collected by ourself, and we now supply it for everyone.
2. the dataset consists of questions in agricultural domain.
3. the dataset is divided into train, dev and test set.

| class	| description |	Train |	Dev |	Test |
| :-----: | :-----------: | :-----: | :---: | :----: |
| 0 |	Ask for material |	12 |	20 |	20 |
| 1 |	Ask for sub region |	259	| 37	| 72 |
| 2 |	Ask for products something could be made |	8 |	13 |	12 |
| 3 |	Ask for product place	| 389 |	55	| 110 |
| 4	| Ask for super classes	| 2589	| 369	| 738 |
| 5	| Ask for  instances of given class	| 194	| 27	| 54 |
| 6	| Ask for son classes	| 8	| 14  |	12 |
| 7	| Ask for locations	| 2231	| 318 |	636 |
| 8	| Others	| 147 |	21 |	40 |


## Code
1. Config.py: the configuration;
2. dowmModel.py: dowmload pre-trained language model on local disk;
3. Model.py: models;
4. main.py: 
