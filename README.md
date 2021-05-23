<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the Global-Temperature-Time-Series and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
*** https://github.com/sn2606/Global-Temperature-Time-Series
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** sn2606, Global-Temperature-Time-Series, https://www.linkedin.com/in/swaranjana-nayak/, swaranjananayak@gmail.com, Global Temperature Time Series Analysis, Singular Spectrum Analysis and ARIMA models implemented on Berkeley Earth Surface Time Series
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/sn2606/Global-Temperature-Time-Series">
  <h1 align="center">Global Temperature Time Series Analysis</h1>
  </a>
  <p>
    Singular Spectrum Analysis and ARIMA models implemented on Berkeley Earth Surface Time Series
    <br />
    <a href="https://github.com/sn2606/Global-Temperature-Time-Series/issues">Report Bugs</a>
  </p>
</p>

#
The Berkeley Earth Surface Temperature Study combines 1.6 billion temperature reports from 16 pre-existing archives. It is nicely packaged and allows for slicing into interesting subsets (for example by country).  Time series analysis is performed on this dataset.

[Link to the dataset](https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data)
#
[Kaggle Notebook for visualizations and ARIMA](https://www.kaggle.com/swaranjananayak/global-temperatures-time-series-analysis)
#
[Kaggle Notebook for SSA](https://www.kaggle.com/swaranjananayak/singular-spectrum-analysis-forecast)
#

## Few points

* [statsmodel.api](https://www.statsmodels.org/stable/index.html) is used for ARIMA implementation
* SSA (decomposition, reconstruction, forecasting) is implemented from scratch as presented in [Golyandina, Nina & Zhigljavsky, Anatoly. (2013). Singular Spectrum Analysis for Time Series. 10.1007/978-3-642-34913-3.](https://www.springer.com/gp/book/9783662624357)
* Python scientific stack is used to simplify all implementations - NumPy, Pandas, SciPy, Seaborn, Matplotlib
* PNG Output of all plots are in the Output folder

## Decompositions and outputs

### Decomposition of time series for ARIMA
![decomp-arima]

### Forecast for ARIMA - mse = 0.09 (on same data for last 12 points i.e. year 2015)
![forecast-arima]

### Decomposition of time series for SSA
![decomp-ssa]

### Forecast for SSA - mse = 0.085 (on same data for last 12 points i.e. year 2015)
![forecast-ssa]

#
<!-- CONTACT -->
## Contact

[@LinkedIn](https://www.linkedin.com/in/swaranjana-nayak/) - swaranjananayak@gmail.com

Project Link: [https://github.com/sn2606/Global-Temperature-Time-Series](https://github.com/sn2606/Global-Temperature-Time-Series)

#
<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* [This Kaggle tutorial notebook](https://www.kaggle.com/jdarcy/introducing-ssa-for-time-series-decomposition)
* [Github Rebository - pssa](https://github.com/aj-cloete/pssa)
* [Deng, Cheng, "Time Series Decomposition Using Singular Spectrum Analysis" (2014). Electronic Theses and Dissertations. Paper 2352. https://dc.etsu.edu/etd/2352](https://dc.etsu.edu/etd/2352/)
* [Golyandina, Nina & Zhigljavsky, Anatoly. (2013). Singular Spectrum Analysis for Time Series. 10.1007/978-3-642-34913-3.](https://www.springer.com/gp/book/9783662624357)
* [ARIMA Model Python Example — Time Series Forecasting](https://towardsdatascience.com/machine-learning-part-19-time-series-and-autoregressive-integrated-moving-average-model-arima-c1005347b0d7)
* [Time Series Data Visulaization with Python](https://machinelearningmastery.com/time-series-data-visualization-with-python/)



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/sn2606/Global-Temperature-Time-Series.svg?style=for-the-badge
[contributors-url]: https://github.com/sn2606/Global-Temperature-Time-Series/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/sn2606/Global-Temperature-Time-Series.svg?style=for-the-badge
[forks-url]: https://github.com/sn2606/Global-Temperature-Time-Series/network/members
[stars-shield]: https://img.shields.io/github/stars/sn2606/Global-Temperature-Time-Series.svg?style=for-the-badge
[stars-url]: https://github.com/sn2606/Global-Temperature-Time-Series/stargazers
[issues-shield]: https://img.shields.io/github/issues/sn2606/Global-Temperature-Time-Series.svg?style=for-the-badge
[issues-url]: https://github.com/sn2606/Global-Temperature-Time-Series/issues
[license-shield]: https://img.shields.io/github/license/sn2606/Global-Temperature-Time-Series.svg?style=for-the-badge
[license-url]: https://github.com/sn2606/Global-Temperature-Time-Series/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/sn2606
[decomp-arima]: Output\decomposition.png
[forecast-arima]: Output\forecast.png
[decomp-ssa]: Output\lat-components-grouped-sep.png
[forecast-ssa]: Output\forecast-ssa.png