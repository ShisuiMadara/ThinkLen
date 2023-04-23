<h1>Description</h1>
Product length estimation using parameters of Title, Product descrption, Bullet points and Product Type Id. This project is made as a part of Amazon ML challenge, all the product title, description and bullet points are the one present in the Amazon shopping application.
<br>
<h1>Technologies used: </h1>
<ul>
  <li>Python</li>
  <li>Tensorflow</li>
  <li>Pytorch</li>
  <li>Sklearn</li>
  <li>CUDA</li>
  <li>Keras</li>
  <li>CuPy</li>
  <li>CuML</li>
  <li>CuDf</li>
</ul>
<br>
<h1>Proposed Solution</h1>
<p>A product in real life is made to cater different problems. Each of the product has different design, use cases and manufacture properties for bridging the gap of the in capabilities of human mind and body.</p>
<p>In the given data, we observed that products which had similar use cases were all about the same dimension. Each of the products having similar properties bore the same “type id”. The similarities could also be extracted using word manipulation and information extraction form the title, product description and bullet points.</p>
<p>The data has a lot of NaN, special characters and different languages. Thus, the data was all translated to English. 
Then for each of the characters, tokens were generation for mapping and easy processing. This data can still contain NaN or null values and thus the values were now removed. </p>
<p>The data was vectorized so as to extract useful information which could have been missed in the case of strings. 
Now, each of the product has a Product_type_id. The similar products (marked by having the same product type id) were clustered together by the use of K mean clustering. </p>
<p>The objects belonging to the same cluster are supposed to have similar dimensions and length and thus, the SVR was trained to predict the product type id based on this assumption. The SVR used RGB kernel and the regularization constant was 100 while the value of epsilon was 0.2. These parameter values were chosen after hypertuning and observations by running on smaller samples of data. </p>
<br>

![WhatsApp Image 2023-04-24 at 00 20 58](https://user-images.githubusercontent.com/77777434/233859848-9a9ea199-51cb-4874-8053-e263c0c24bc9.jpeg)

