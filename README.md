# Drinkable Water Analyser

In Epitech, Tek4 students have the opportunity to travel around the world.
It is an incredible opportunity to learn new things and discover the world at the same time.
However, the water quality is not the same everywhere, depending on the health regulation.
Therefore it is important to know quickly if the water at the students' disposal is healthy to drink or not!

<p align="center">
<img src="https://domf5oio6qrcr.cloudfront.net/medialibrary/7909/b8a1309a-ba53-48c7-bca3-9c36aab2338a.jpg" />
</p>

This little Web App will allow people to quickly check if the water is drinkable or not, as long as they have data regarding their water.
Using a trained Artificial Intelligence, our application will quickly be able to analyse the potability of the water with given parameters. 

# [Dataset](https://www.kaggle.com/adityakadiwal/water-potability)

Here is the link to the dataset used to train our Artificial Intelligence.
It contains samples of 3,276 different water bodies.

### [Attributes information:]()

* **pH**: The Acid-Base property of the water. Recommended value is between 6.5 and 8.5.
* **Hardness** : Concentration of calcium and magnesium salts.
* **Solids** : The ability for the water to dissolve solids (minerals).
* **Chloramines** : Water disinfectants concentration (recommended amount is 4mg/L).
* **Sulfate** : Sulfates is present is most minerals, soil and rocks.
* **Conductivity** : Electrical conductivity measures the ionic concentration (should not exceed 400 μS/cm).
* **Organic Carbon** : Total Organic Carbon represents the natural decaying organic matter organic matter and synthetic sources.
* **Trihalomethanes** : THMs can be found in water treated with chlorine.
* **Turbidity** : The turbidity of water depends on the quantity of solid matter present in the suspended state.

### [Experiment Results:]()
* **Data Analysis**
  * Some values may contains NaN/Null values, but they are handled.
* **Performance Evaluation**
  * Splitting the dataset by 75% for training set and 25% validation set.
* **Training and Validation**
  * RandomForest and GradientBoosting get a higher accuracy score than other classification models.
    * RandomForest : 77%
    * GrandientBoosting : 77.7%
* **Performance Results**
  * Training Score : 100.0%
  * Validation Score : ~80%
  * Results displayed above are considered satisfying, because we biased the training and test to make sure the analysis handles every possible case.

# [Demo](https://share.streamlit.io/tim-snugget/ml_drinkable_water/main/app.py)
Live Demo : https://share.streamlit.io/tim-snugget/ml_drinkable_water/main/app.py

![](https://media.discordapp.net/attachments/897066717052801037/904338073209430057/unknown.png?width=577&height=676)
![](https://media.discordapp.net/attachments/897066717052801037/904338102548582410/unknown.png?width=559&height=676)

# [Github and Contributors](https://github.com/Tim-Snugget/ML_Drinkable_Water)
* [Aldric Liottier](https://github.com/aldricLiottier)
* [Pierre Marcurat](https://github.com/3uph0riah)
* [Timothé "Tim-Snugget" Fertin](https://github.com/Tim-Snugget)

# References
EduFlow
https://www.kaggle.com/adityakadiwal/water-potability
https://www.youtube.com/watch?v=aircAruvnKk
