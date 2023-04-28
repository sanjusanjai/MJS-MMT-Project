# MJS-MMT-Project

`billboard_v{1,2}.py` generates csv files of hitsongs from billboard for the given years
<br>
`data_v{1,2,3}.py` makes then given data.txt file into a csv file.
<br>
`english_v1.py` trains model for the english data obtained
<br>
`neuralnetwork_v1.py` trains different models for the data. parameters and model can be selected from the `main()` function.
<br>
`parametertuning.py` creates multiple models with diff parameters and reports the parameters with best accuracy.
<br>
`svm_v{1,2,3}.py` trains svm mdoels.
<br>
`weights.ipynb` trains a model and checks the weights of the first layer to see which feature has more weight.
<br>
`wiki*.py` extracts scrapes data from wikipedia for hitsongs.
<br>
`./dataCollection/extract_v3.py` crawls the Indian hit songs website to get the song details for each year and outputs the song information crawled from the website, using BeautifulSoup package, to a csv file for each year.
<br>
`./dataCollection/mp3.py` Uses the above generated csv files for each year to get views of each song using the corresponding youtube link to sort the songs. Once songs were sorted, I used the youtube-dl package to check the duration of each song if it was lesser than 2 minutes, then that song was removed from the list and this sorted list of dictionaries was saved to local storage as getting youtube views for each song and checking song duration took a surprisingly long time. Using this sorted list of songs and links, 30 top and bottom songs were downloaded from each year using youtube-dl package and trimmed to include only the middle 2 minutes of the song.
<br>
`./dataCollection/ytViews.sh` Bash script to extract the views of a song from the youtube page for the corresponding song.

