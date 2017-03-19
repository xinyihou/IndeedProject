
<h1> <center> Job Posting Analyis :Data Scientist Vs Software engineer from Cybercoder <center>
</h1>
<img width=200 height="200" src="http://assets1.csc.com/big_data/images/analytics_glasses-109438991.jpg">

<p>
&emsp;&emsp; Nowadays, the data science field is hot, and it is unlikely that this will change in the near future. While a data driven approach is finding its way into all facets of business, companies are fiercely fighting for the best data analytic skills that are available in the market, and salaries for data science roles are going in overdrive. Compare with the commonly popular IT related job position, such as software engineer, most of big companies’ increased focus on acquiring data science talent goes hand in hand with the creation of a whole new set of data science roles and titles. 
</p>

<p>&emsp;&emsp;On the other hand, we are all graduating this spring with the degree of statistics. So we are interested in the job placement for the statistic degree. As we discussed above, the data scientists and analysts are really popular in the job market. We wonder the difference of data scientist and software engineers in term of location, salary, skill sets, experience, degree preference. So we want to find a online employment search website to gather the in-time information and data to figure out this problem. 
</p>

<p>&emsp;&emsp; <a href="https://www.cybercoders.com/">CyberCoders</a> is one of the innovative employment search website in the state. The version of cybercoder’s website is really clear and formatted. Since their posts have no outside links like other employment search websites, we are easier to get the content of each post to construct a data frame. Also, this website focuses more on the IT related job markets, so it is perfect for us to analyze content. Additionally, this website is well organized and frequently update since we found the most of job are posted within 10 days.  
</p>

<center><img width=700 height=200 src="https://github.com/xinyihou/IndeedProject/blob/master/Screen%20Shot%202017-03-07%20at%203.39.04%20PM.png?raw=true" alt="Main search page from Cybercoder"/> <center> <br>

<center><img width=70 height=70 src="http://www.javaww.com/images/arrow_down.png"><center> <br>

<center><img width=700 height=200 src ="https://github.com/xinyihou/IndeedProject/blob/master/edit%20data%20frame.jpg?raw=true"> <br><center>

<p><center> **The original parser program is <a href="https://github.com/xinyihou/IndeedProject/blob/master/Project_edited.ipynb">here</a>.**
<center>
</p>
<p>&emsp; </p>

<p> &emsp;&emsp;In our project, we get the information of 109 Data Scientist and 200 Software Engineer job postings on CyberCoders through web scraping, which includes the job title, id, description, post data, salary range, preferred skills, city, and state. We compare the salary of DS and SDE, also including the comparison among different part of US. What is more, we find the need of years of experience through regular expression, the most important skills through NLP techniques. The degree required for the job and the posting dates are also topics we are interested in.
</p>


```python
from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')
```




<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>




```python
ds = pd.read_csv('data scientist.csv',index_col=False)
del ds['Unnamed: 0']
```


```python
import numpy as np
import pandas as pd
import nltk
import string
import unicodedata
from collections import Counter
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import os,re
from collections import Counter
from datetime import datetime,date
import seaborn as sns
import folium
from IPython.display import HTML
from IPython.display import IFrame
import statsmodels.api as sm
from statsmodels.formula.api import ols
from itertools import compress
from nltk.corpus import stopwords
from wordcloud import WordCloud,STOPWORDS
```


```python
ds = pd.read_csv('data scientist.csv',index_col=False)
del ds['Unnamed: 0']
```

<h3><center>Posting Dates<center><h3/>



```python
print Counter(ds['post_date'])
```

    Counter({'3/4/17': 72, '2/23/17': 21, '2/28/17': 3, '3/1/17': 2, '3/3/17': 2, '1/30/17': 1, '1/4/17': 1, '2/2/17': 1, '2/17/17': 1, '1/9/17': 1, '1/17/17': 1, '12/14/16': 1, '2/20/17': 1, '2/3/17': 1})


<p> &emsp;&emsp; We get these data on Mar 4th 2017, and we want to know the posting date of these jobs. Here, you can find a interesting thing. The post dates are not uniformly distributed, and most of jobs are posted on 3/4/2017 and 2/23/2017. If you open the CyberCoders web now(3/5/2017), you can find a lot of jobs, whose job id is same as the ones of yesterday, are marked as 'Posting Today'.</p>


```python
datevalues = Counter(ds['post_date'])
datevalues = [datetime.strptime(i,'%m/%d/%Y') for i in datevalues]
[datetime(2017,3,4)-i for i in datevalues]
ds['post_date'] = pd.to_datetime(ds['post_date'])
plot = sns.factorplot('post_date',kind = 'count',data = ds,size=4, aspect=2)
plot.set(xticklabels=['12/14/2016','','','','','','','','','02/23/2017','','','','03/04/2017'])
plt.title('Job Posting Numbers vs Dates')
plt.show()
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-65-62bdbdaf0151> in <module>()
          1 datevalues = Counter(ds['post_date'])
    ----> 2 datevalues = [datetime.strptime(i,'%m/%d/%Y') for i in datevalues]
          3 [datetime(2017,3,4)-i for i in datevalues]
          4 ds['post_date'] = pd.to_datetime(ds['post_date'])
          5 plot = sns.factorplot('post_date',kind = 'count',data = ds,size=4, aspect=2)


    /usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/python2.7/_strptime.pyc in _strptime(data_string, format)
        330     if not found:
        331         raise ValueError("time data %r does not match format %r" %
    --> 332                          (data_string, format))
        333     if len(data_string) != found.end():
        334         raise ValueError("unconverted data remains: %s" %


    ValueError: time data '1/30/17' does not match format '%m/%d/%Y'


<p> &emsp;&emsp; The oldest job is posted on 12/14/2016, 80 days ago. However, most jobs are posted in recent 10 days.</p>


```python
ds['post_date'] = pd.to_datetime(ds['post_date'])
plot = sns.factorplot('post_date',kind = 'count',data = ds,size=4, aspect=2)
plot.set(xticklabels=['12/14/2016','','','','','','','','','02/23/2017','','','','03/04/2017'])
plt.title('Job Posting Numbers vs Dates')
plt.show()
```


![png](output_10_0.png)


<h3><center>**Location**<center><h3/>


```python
from geopy.geocoders import Nominatim
geolocator = Nominatim()
loc = geolocator.geocode("New York, NY")
loc
ds['location'] = ds['city']+','+ds['state']
lonlat = [geolocator.geocode(i) for i in ds.location]
lonlat[2]
print Counter(ds['state'])
```


    ---------------------------------------------------------------------------

    GeocoderTimedOut                          Traceback (most recent call last)

    <ipython-input-67-43cebe5a4a90> in <module>()
          4 loc
          5 ds['location'] = ds['city']+','+ds['state']
    ----> 6 lonlat = [geolocator.geocode(i) for i in ds.location]
          7 lonlat[2]
          8 print Counter(ds['state'])


    /usr/local/lib/python2.7/site-packages/geopy/geocoders/osm.pyc in geocode(self, query, exactly_one, timeout, addressdetails, language, geometry)
        191         logger.debug("%s.geocode: %s", self.__class__.__name__, url)
        192         return self._parse_json(
    --> 193             self._call_geocoder(url, timeout=timeout), exactly_one
        194         )
        195 


    /usr/local/lib/python2.7/site-packages/geopy/geocoders/base.pyc in _call_geocoder(self, url, timeout, raw, requester, deserializer, **kwargs)
        161             elif isinstance(error, URLError):
        162                 if "timed out" in message:
    --> 163                     raise GeocoderTimedOut('Service timed out')
        164                 elif "unreachable" in message:
        165                     raise GeocoderUnavailable('Service not available')


    GeocoderTimedOut: Service timed out


We transform the text to the real GPS data. Above location GPS data is the one example of how we get.  And we can see the count of the number of job post of each state. 


```python
mapds = folium.Map(location=[39,-98.35], zoom_start=4)
marker_cluster = folium.MarkerCluster("Data Scientist Job").add_to(mapds)
for each in lonlat:
    folium.Marker(each[1]).add_to(marker_cluster)
    folium.MarkerCluster()
mapds.save('map.html')
IFrame('map.html', width=800, height=500)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-68-88156f73dfe4> in <module>()
          1 mapds = folium.Map(location=[39,-98.35], zoom_start=4)
          2 marker_cluster = folium.MarkerCluster("Data Scientist Job").add_to(mapds)
    ----> 3 for each in lonlat:
          4     folium.Marker(each[1]).add_to(marker_cluster)
          5     folium.MarkerCluster()


    NameError: name 'lonlat' is not defined


<h3><center>Salary<center><h3/>



```python
sum(pd.isnull(ds['salary_lower']))
ds2 = ds[pd.notnull(ds['salary_lower'])].copy()

ds2 = ds2[ds2.salary_lower>0]
#Only 74 records now
ds2['salary_mid']=(ds.salary_lower+ds.salary_upper)/2
print Counter(ds2.state)
```

    Counter({'CA': 36, 'NY': 11, 'MA': 7, 'WA': 6, 'MD': 2, 'IL': 2, 'CT': 2, 'TX': 1, 'OH': 1, 'CO': 1, 'VA': 1, 'PA': 1, 'SC': 1, 'MO': 1, 'AZ': 1, 'OR': 1})


In the 109 data scientist posts we got from the cybercoder, there are 31 post without specific salary range, which denotes as unspecified. Also, there are 74 posts with positive salary range.


```python
d={}
d['east']=['CT','MA','MD','NY','PA','SC','VA','ME','VT','NH','RI','NJ','DE','WV','NC','GA','AL']
d['west']=['CA','OR','WA','AK','MO','ID','MT','NV','UT','WY']
d['other']=['AZ','CO','IL','OH','TX']

ds2['part']=''

index = [i in d['east'] for i in ds2.state]
index2 = [i in d['west'] for i in ds2.state]
index3 = [i in d['other'] for i in ds2.state]
ds2.loc[index,'part']='east'
ds2.loc[index2,'part']='west'
ds2.loc[index3,'part']='other'

Counter(ds2.part)

ds2.boxplot("salary_mid", "part")
plt.show()


```


![png](output_18_0.png)


The above box plot shows that the west coast has the highest salary median among the whole state. The result makes sense since California have the Silicon Valley which aggregates a crowd of the most professional data scientist compared with other place. Also, we fit the linear regression model for the salary median and the states. 


```python
mod = ols('salary_mid ~ part',
                data=ds2).fit()
                
aov_table = sm.stats.anova_lm(mod, typ=2)
print aov_table

```

                    sum_sq    df         F    PR(>F)
    part      1.344950e+10   2.0  5.202564  0.007756
    Residual  9.306600e+10  72.0       NaN       NaN


<center>** What's the situation of the software development engineer?**


```python
sde = pd.read_csv('Software_Engineer.csv',index_col=False)
del sde['Unnamed: 0']

sde2 = sde[pd.notnull(sde['salary_lower'])].copy()

sde2 = sde2[sde2.salary_lower>0]

print len(sde)
print len(sde2)
```

    200
    157


<p> There are 200 job posts of software development engineers in the website and just 157 posts with a positive salary range.


```python
sde2['salary_mid']=(sde2.salary_lower+sde2.salary_upper)/2
Counter(sde2.state)
sde2['part']='other'
index = [i in d['east'] for i in sde2.state]
index2 = [i in d['west'] for i in sde2.state]
sde2.loc[index,'part']='east'
sde2.loc[index2,'part']='west'
ds2['type']='Data Scientist'
sde2['type']='Software Engineer'
dssde = ds2.append(sde2)
dssde.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>job_id</th>
      <th>location</th>
      <th>need_for_position</th>
      <th>part</th>
      <th>post_date</th>
      <th>preferred_skill</th>
      <th>salary_lower</th>
      <th>salary_mid</th>
      <th>salary_upper</th>
      <th>state</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Newton</td>
      <td>BA-1277535</td>
      <td>Newton,MA</td>
      <td>- BS (min GPA 3.5) or MS or PhD in science, en...</td>
      <td>east</td>
      <td>2017-02-23 00:00:00</td>
      <td>Data Analytics, Informatics, Life Sciences . P...</td>
      <td>100000.0</td>
      <td>115000.0</td>
      <td>130000.0</td>
      <td>MA</td>
      <td>Data Scientist</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sunnyvale</td>
      <td>BF1-1327877</td>
      <td>Sunnyvale,CA</td>
      <td>- Networking/Security  - Experience with big d...</td>
      <td>west</td>
      <td>2017-02-23 00:00:00</td>
      <td>Python, C/C++, Networking, Security, Apache Sp...</td>
      <td>150000.0</td>
      <td>175000.0</td>
      <td>200000.0</td>
      <td>CA</td>
      <td>Data Scientist</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Redwood City</td>
      <td>AW2-1341356</td>
      <td>Redwood City,CA</td>
      <td>Requirements: Bachelors in Computer Science or...</td>
      <td>west</td>
      <td>2017-02-23 00:00:00</td>
      <td>Machine Learning, Python, R, Mapreduce, Javasc...</td>
      <td>140000.0</td>
      <td>182500.0</td>
      <td>225000.0</td>
      <td>CA</td>
      <td>Data Scientist</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Portland</td>
      <td>CS9-1346787</td>
      <td>Portland,OR</td>
      <td>Experience and knowledge of: - Machine Learnin...</td>
      <td>west</td>
      <td>2017-02-23 00:00:00</td>
      <td>Machine Learning, Data Mining, Python, ETL BI,...</td>
      <td>100000.0</td>
      <td>110000.0</td>
      <td>120000.0</td>
      <td>OR</td>
      <td>Data Scientist</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Needham</td>
      <td>PD2-1346845</td>
      <td>Needham,MA</td>
      <td>- BS with a focus on life sciences. A degree i...</td>
      <td>east</td>
      <td>2017-02-23 00:00:00</td>
      <td>Data Analytics, Life Sciences, Pharmaceuticals...</td>
      <td>100000.0</td>
      <td>115000.0</td>
      <td>130000.0</td>
      <td>MA</td>
      <td>Data Scientist</td>
    </tr>
  </tbody>
</table>
</div>



<p> We use the same method on SDE that we apply into the data scientist. Then the above dataframe is the combination of the data scientist and SDE.</p>


```python
sns.set(rc={"figure.figsize": (8, 4)})
sns.distplot(dssde.salary_mid[dssde['type']=='Data Scientist'],hist_kws={"label":'DS'})
sns.distplot(dssde.salary_mid[dssde['type']=='Software Engineer'],hist_kws={"label":'SDE'})
plt.title('Distribution of Salary: DS vs SDE')
plt.legend()
plt.show()
```


![png](output_26_0.png)


The above plot shows the distribution of salary between data scientist and SDE. Actually, the  salary median of SDE is higher than data scientist. 


```python
sns.boxplot(x="part", y="salary_mid", hue="type", data=dssde,palette="Set1")
plt.title('Salary Compare: DS VS SDE')
plt.show()

mod2 = ols('salary_mid ~ part+type',
                data=dssde).fit()
                
aov_table2 = sm.stats.anova_lm(mod2, typ=2)
print aov_table2

print mod2.summary()
```


![png](output_28_0.png)


                    sum_sq     df          F        PR(>F)
    part      2.344070e+10    2.0  13.568313  2.706902e-06
    type      5.785471e+10    1.0  66.976735  1.936368e-14
    Residual  1.969471e+11  228.0        NaN           NaN
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:             salary_mid   R-squared:                       0.355
    Model:                            OLS   Adj. R-squared:                  0.346
    Method:                 Least Squares   F-statistic:                     41.75
    Date:                Sat, 18 Mar 2017   Prob (F-statistic):           1.53e-21
    Time:                        17:26:08   Log-Likelihood:                -2714.1
    No. Observations:                 232   AIC:                             5436.
    Df Residuals:                     228   BIC:                             5450.
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    =============================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------------
    Intercept                  1.435e+05   4398.289     32.627      0.000    1.35e+05    1.52e+05
    part[T.other]             -1.199e+04   5318.300     -2.255      0.025   -2.25e+04   -1512.390
    part[T.west]               1.459e+04   4416.796      3.302      0.001    5882.069    2.33e+04
    type[T.Software Engineer] -3.501e+04   4278.214     -8.184      0.000   -4.34e+04   -2.66e+04
    ==============================================================================
    Omnibus:                       46.401   Durbin-Watson:                   1.930
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               93.867
    Skew:                           0.984   Prob(JB):                     4.14e-21
    Kurtosis:                       5.416   Cond. No.                         4.59
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


<p> From the location view, we can see that two types of job in the west coast are still higher than other place and the salary of SDE is still higher than Data Scientist.

<h3><center>Experience<center></h3>

<p>We want to know how many of the job postings specify the exact number of years of experience. We use regular expression to get this kind of info.<p>



```python
ds.need_for_position = [i.lower() for i in ds.need_for_position] 
yoe= [re.findall(r'[0-9\-\\+0-9]+ years of ',i) for i in ds.need_for_position]
yoe[:5]
len(ds.need_for_position)- sum(i==[] for i in yoe)
```




    51



<p>Among the 109 jobs, 49 of them specify the years of experience.</p>


```python
yoe2 = list(compress(yoe, [i!=[] for i in yoe]))
del yoe2[8]
del yoe2[17]
yoe3 = [int(i[0][0]) for i in yoe2]
Counter(yoe3).keys()
Counter(yoe3).values()
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-77-aa278090eae7> in <module>()
          2 del yoe2[8]
          3 del yoe2[17]
    ----> 4 yoe3 = [int(i[0][0]) for i in yoe2]
          5 Counter(yoe3).keys()
          6 Counter(yoe3).values()


    ValueError: invalid literal for int() with base 10: '-'



```python
plt.bar(Counter(yoe3).keys(),Counter(yoe3).values())
plt.xlabel('Years of experience')
plt.ylabel('Count')
plt.title('Bar Plot of Years of Experience-Data Scientist')
plt.show()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-78-65e638e05a51> in <module>()
    ----> 1 plt.bar(Counter(yoe3).keys(),Counter(yoe3).values())
          2 plt.xlabel('Years of experience')
          3 plt.ylabel('Count')
          4 plt.title('Bar Plot of Years of Experience-Data Scientist')
          5 plt.show()


    NameError: name 'yoe3' is not defined


<p> From the pie plot, we know that most of the job required 3 years experience before you apply for the job. This also denotes that the hard situation of finding the job in today's IT related job market. 

<h3><center>Skill Set<center></h3>



```python
ds_skill =",".join( ds['preferred_skill'] ).lower()
ds_needForPosition ="".join( ds['need_for_position']).lower()
def tokenize(text):
    s = text.lower()
    s = re.sub(r'/|\(|\)', ',', s.lower()).split(',')
    s = [i.strip() for i in s if i != '']
    return s
```


```python
# skill set from prefered_skill ('sql' vs 'sql database', )
ds_filtered_skill = [word for word in tokenize(ds_skill) if word not in stopwords.words('english')] 
nltk.FreqDist(ds_filtered_skill).plot(30)
```


![png](output_38_0.png)


<p> From the above plot, we can see that the Python was the top one among the preferred skills, which means that STA 141 is a really useful class for us entering into the job market.</p>


```python
# Two sets of words with intersection
ds_skill_words = pd.DataFrame(nltk.FreqDist(ds_filtered_skill).most_common(8) )
ds_skill_words.iloc[:,1] = ds_skill_words.iloc[:,1] / ds.shape[0] 
ds_barplot = sns.barplot( x = 0, y = 1,data = ds_skill_words, palette = "Blues_d")
ds_barplot.set(xlabel = '', ylabel = 'percentage in D.S. posts')
plt.show()
```


![png](output_40_0.png)



```python
print ds_filtered_skill
skill = ds_filtered_skill[:]
for n, i in enumerate(skill):
    if i == 'r':
        skill[n] = 'R+++'
print  skill
```

    ['data analytics', 'informatics', 'life sciences . pharmaceutical industry', 'java', 'python', 'python', 'c', 'c++', 'networking', 'security', 'apache spark', 'kafka', 'elasticsearch', 'mongodb', 'big data', 'predictive modeling', 'algorithm development', 'data mining', 'random forest', 'bayesian networks', 'bayesian modeling', 'markov chains', 'nosql databases', 'sql databases', 'machine learning', 'python', 'r', 'mapreduce', 'javascrip', 'spark', 'streaming', 'machine learning', 'data mining', 'python', 'etl bi', 'and data pipelines', 'r', 'hadoop', 'advanced statistical analysis', 'data analytics', 'life sciences', 'pharmaceuticals', 'java', 'python', 'informatics', 'big data', 'predictive modeling', 'algorithm development', 'data mining', 'random forest', 'bayesian networks', 'bayesian modeling', 'markov chains', 'nosql databases', 'sql databases', 'machine learning', 'python', 'linux', 'java', 'scala', 'r', 'spark', 'machine learning', 'python', 'linux', 'java', 'scala', 'r', 'spark', 'machine learning', 'graph analytics', 'statistical modeling', 'data analytics', 'python', 'r', 'hadoop', 'aws', 'sql', 'hive', 'python', 'r', 'sas', 'pandas', 'data mining', 'unix', 'linux environments', 'big data', 'cloud', 'openstack', 'hadoop', 'solr', 'hbase', 'spark', 'docker', 'ansible', 'spring', 'data science', 'python', 'scikit.learn', 'virtual environments', 'bayesian inference', 'machine learning algorithms', 'teaching experience', 'machine learning', 'python', 'linux', 'java', 'scala', 'r', 'spark', 'java', 'python', 'scala', 'r', 'mapreduce', 'hive', 'spark', 'machine learning', 'python', 'data mining', 'hadoop', 'scala', 'spark', 'sql', 'r', 'python', 'sql', 'hadoop', 'algorithms', 'machine learning', 'data mining', 'python', 'machine learning', 'hadoop', 'aws', 'data mining', 'r', 'python', 'sql', 'hadoop', 'algorithms', 'machine learning', 'data mining', 'data analytics', 'healthcare industry', 'data science', 'sql or mysql', 'sas', 'statistical software', 'r', 'python', 'sas', 'spss', 'sql', 'r', 'python', 'sql', 'hadoop', 'algorithms', 'machine learning', 'data mining', 'machine learning', 'artificial intelligence', 'statistics', 'data analysis', 'unix', 'linux', 'statistical software', 'machine learning', 'data mining', 'python', 'hadoop', 'machine learning', 'python', 'data mining', 'r', 'phd preferred', 'machine learning', 'spark', 'java', 'python', 'scala', 'data science', 'from industry', 'machine learning algorithms', 'python', 'phenomenal written and oral communication', 'data manipulation', 'mining', 'modeling', 'machine learning', 'natural language processing', 'large data sets', 'hadoop', 'hands on team leadership', 'python', 'r', 'python', 'sql', 'hadoop', 'algorithms', 'machine learning', 'data mining', 'machine learning', 'data mining', 'hadoop', 'presto', 'sql server', 'mysql', 'machine learning', 'data mining', 'python', 'hadoop', 'scala', 'spark', 'sql', 'python', 'python scientific stack', 'numpy', 'pandas', 'scipy', 'etc.', 'basic statistics and data analysis', 'open source software', 'data ingestion and processing', 'machine learning', 'analytics experience', 'financial industry a plus', 'agile framework with atlassian', 'or similar', 'unclean', 'semi- structured', 'unstructured data', 'unsecured lending credit experience', 'python', 'basic statistics and data analysis', 'open source software', 'data ingestion and processing', 'machine learning', 'analytics experience', 'financial industry a plus', 'agile framework with atlassian', 'or similar', 'unclean', 'semi- structured', 'unstructured data', 'credit modeling', 'hadoop', 'spark', 'h20', 'data science', 'python', 'scikit.learn', 'virtual environments', 'bayesian inference', 'machine learning algorithms', 'prior teaching experience', 'not essential', 'tcp', 'ip', 'machine learning', 'data mining', 'hadoop', 'hive', 'cassandra', 'storm', 'or spark', 'information retrieval', 'regression', 'support vector machines', 'optimization', 'machine learning', 'artificial intelligence', 'statistics', 'data analysis', 'unix', 'linux', 'statistical software', 'data science', 'artificial intelligence', 'machine learning', 'masters or phd', 'cs', 'ai', 'machine learning', 'statistical analysis', 'open source machine', 'deep learning frameworks', 'pandas', 'python', 'numpy', 'jupyter', 'predictive modeling', 'hadoop', 'spark', 'mongodb', 'ruby', 'python', 'cyber security', 'scala', 'spark', 'machine learning', 'java', 'python', 'hadoop', 'scala', 'spark', 'machine learning', 'java', 'python', 'hadoop', 'python', 'c', 'c++', 'networking', 'security', 'apache spark', 'kafka', 'elasticsearch', 'mongodb', 'python', 'c', 'c++', 'networking', 'security', 'apache spark', 'kafka', 'elasticsearch', 'mongodb', 'data scientist', 'python', 'r', 'recommendation systems', 'machine learning', 'sql', 'python', 'pandas', 'jupyter', 'numpy', 'analytics', 'machine learning', 'text-mining', 'nlp', 'pathway', 'network analysis', 'modeling & simulations', 'java or c#', 'big data', 'cloud computing', 'pandas', 'python', 'numpy', 'jupyter', 'machine learning', 'python', 'r', 'sql', 'bi reporting', 'tableau', 'obiee', 'business objects', 'congos', 'micro strategy', 'sas', 'machine learning', 'python', 'r', 'sql', 'bi reporting', 'tableau', 'obiee', 'business objects', 'congos', 'micro strategy', 'sas', 'machine learning', 'deep learning', 'nlp', 'phd', 'ms in computer science', 'r', 'python', 'sql', 'hadoop', 'algorithms', 'machine learning', 'data mining', 'algorithm development', 'bi reporting tools', 'python', 'statistical modeling', 'online advertising', 'data pipeline architecture', 'statistical analysis & programming', 'redshift', 'sql', 'r', 'matlab', 'python', 'data mining', 'data analytics & visualization', 'python', 'matlab', 'r', 'python', 'aws', 'data science', 'data science', 'python', 'scikit.learn', 'virtual environments', 'bayesian inference', 'machine learning algorithms', 'prior teaching experience', 'not essential', 'json', 'machine learning', 'nlp', 'artificial intelligence', 'data transfer', 'chief scientist', 'solutions architect', 'big data', 'r or python', 'algorithm development', 'bi reporting tools', 'python', 'statistical modeling', 'online advertising', 'machine learning', 'data analysis', 'big data', 'data visualization', 'd3', 'predictive modeling', 'predictive analytics', 'data science', 'python', 'scikit.learn', 'virtual environments', 'bayesian inference', 'machine learning algorithms', 'teaching experience', 'r', 'python', 'sql', 'hadoop', 'algorithms', 'machine learning', 'data mining', 'machine learning', 'statistical modeling', 'data analytics', 'python', 'hadoop', 'aws', 'spark', 'mission critical systems', 'machine learning', 'phd', 'field analysis', 'python', 'r', 'matlab', 'sas', 'machine learning', 'python', 'linux', 'java', 'scala', 'r', 'spark', 'r', 'python', 'sql', 'hadoop', 'algorithms', 'machine learning', 'data mining', 'data science', 'from industry', 'machine learning algorithms', 'python', 'phenomenal written and oral communication', 'r', 'python', 'sql', 'aws', 'hadoop', 'predictive modeling', 'algorithm development', 'bi reporting tools', 'python', 'statistical modeling', 'online advertising', 'data science', 'python', 'scikit.learn', 'virtual environments', 'bayesian inference', 'machine learning algorithms', 'teaching experience', 'r', 'python', 'sql', 'hadoop', 'algorithms', 'machine learning', 'data mining', 'machine learning', 'java', 'scala', 'spark', 'python', 'data mining', 'r', 'bioinformatics', 'python', 'perl', 'java', 'r', 'python', 'sql', 'hadoop', 'algorithms', 'machine learning', 'data mining', 'machine learning', 'python', 'r', 'sql', 'bi reporting', 'tableau', 'obiee', 'business objects', 'congos', 'micro strategy', 'sas', 'r', 'matlab', 'cnn', 'rnn', 'python', 'scala', 'or java', 'gpu programming', 'big data', 'spark', 'data science', 'artificial intelligence', 'ai', 'machine learning', 'deep learning', 'robotics', 'security data', 'manufacturing data', 'healthcare data', 'machine learning', 'python', 'linux', 'java', 'scala', 'r', 'spark', 'statistics', 'spss', 'sas', 's-plus', 'r', 'analyzing large data sets', 'machine learning', 'clustering & classification', 'nlp', 'optimization algorithms', 'causal inference', 'deep learning', 'predictive modeling concepts', 'experiment design', 'computer vision', 'machine learning', 'python', 'scala', 'or java', 'hadoop', 'big data', 'r', 'matlab', 'c++', 'scala', 'spark', 'machine learning', 'java', 'python', 'hadoop', 'machine learning', 'hadoop', 'python', 'raw data analysis', 'data visualization', 'statistics', 'data manipulation', 'sql', 'r', 'tableau', 'data science', 'management', 'leadership', 'creativity', 'curiosity', 'machine learning', 'predictive modeling techniques', 'python', 'r', 'data visualization', 'game theory', 'machine learning', 'python', 'applied mathematics', 'r', 'software development', 'data scientist', 'deep learning', 'machine learning', 'computer vision', 'algorithm dev', 'python', 'java', 'hadoop', 'data scientist', 'python', 'hadoop', 'predictive analystics', 'novel machine learning', 'algorithms', 'ts', 'sci polygraph clearance', 'federal based engineering', 'c', 'c++', 'java', 'hadoop', 'python', 'numpy', 'pandas', 'matplotlib', 'd3js', 'tableau', 'machine learning', 'data mining', 'spss', 'sas', 'r', 'matlab', 'celect', 'datarobot', 'netica', 'data modeling', 'data warehousing', 'big data', 'hadoop', 'python', 'numpy', 'pandas', 'matplotlib', 'd3js', 'tableau', 'machine learning', 'data mining', 'data scientist', 'machine learning', 'r', 'numpy', 'spicy', 'pandas', 'hadoop mapreduce', 'apache spark', 'hive', 'probability theory and mathematical optimization', 'java', 'c++', 'unix', 'linux', 'machine learning', 'computer vision', 'caffe', 'knn', 'deep learning', 'data science', 'quantitative finance', 'systematic trading', 'proprietary trading', 'factor models', 'machine learning', 'machine learning', 'hadoop', 'python', 'raw data analysis', 'data visualization', 'statistics', 'data manipulation', 'sql', 'r', 'python', 'sql', 'data engineering', 'aws', 'machine learning', 'deep learning', 'nlp', 'phd', 'ms in computer science', 'data science', 'machine learning', 'sas', 'spss', 'r', 'python', 'statistical analysis', 'data science', 'machine learning', 'data mining', 'sql', 'python', 'scala', 'or java', 'distributed frameworks', 'machine learning', 'algorithms', 'financial statements', 'regression analysis', 'cluster analysis', 'r', 'sql', 'data mining', 'ph.d', 'data science', 'machine learning', 'data mining', 'sql', 'python', 'scala', 'or java', 'distributed frameworks', 'java', 'java springsource', 'c++', 'c#', 'data mining', 'machine learning', 'r', 'neo4j', 'rest-based apis', 'machine learning', 'algorithms', 'financial statements', 'regression analysis', 'cluster analysis', 'r', 'sql', 'data mining', 'ph.d', 'r', 'python', 'scikit learn', 'numpy', 'visual intelligence', 'aws', 'data science', 'clinical laboratory scientist', 'ascp', 'toxicology', 'data analysis', 'data manipulation', 'mining', 'modeling', 'machine learning', 'natural language processing', 'large data sets', 'hadoop', 'hands on team leadership', 'python', 'machine learning', 'graph analytics', 'statistical modeling', 'data analytics', 'python', 'r', 'hadoop', 'aws', 'food', 'microbiology', 'data compilation', 'data analysis', 'leadership', 'genome sequencing', 'proposals and reports', 'staffing', 'experiment execution', 'research scientist', 'machine learning', 'deep learning', 'nlp', 'artificial intelligence', 'neural networks', 'research scientist', 'machine learning', 'deep learning', 'nlp', 'artificial intelligence', 'neural networks', 'clustering', 'pattern recognition', 'nlp', 'mathematical', 'statistical modeling', 'natural language processing', 'neural networks', 'java', 'data mining', 'chief scientist', 'solutions architect', 'big data', 'r or python']
    ['data analytics', 'informatics', 'life sciences . pharmaceutical industry', 'java', 'python', 'python', 'c', 'c++', 'networking', 'security', 'apache spark', 'kafka', 'elasticsearch', 'mongodb', 'big data', 'predictive modeling', 'algorithm development', 'data mining', 'random forest', 'bayesian networks', 'bayesian modeling', 'markov chains', 'nosql databases', 'sql databases', 'machine learning', 'python', 'R+++', 'mapreduce', 'javascrip', 'spark', 'streaming', 'machine learning', 'data mining', 'python', 'etl bi', 'and data pipelines', 'R+++', 'hadoop', 'advanced statistical analysis', 'data analytics', 'life sciences', 'pharmaceuticals', 'java', 'python', 'informatics', 'big data', 'predictive modeling', 'algorithm development', 'data mining', 'random forest', 'bayesian networks', 'bayesian modeling', 'markov chains', 'nosql databases', 'sql databases', 'machine learning', 'python', 'linux', 'java', 'scala', 'R+++', 'spark', 'machine learning', 'python', 'linux', 'java', 'scala', 'R+++', 'spark', 'machine learning', 'graph analytics', 'statistical modeling', 'data analytics', 'python', 'R+++', 'hadoop', 'aws', 'sql', 'hive', 'python', 'R+++', 'sas', 'pandas', 'data mining', 'unix', 'linux environments', 'big data', 'cloud', 'openstack', 'hadoop', 'solr', 'hbase', 'spark', 'docker', 'ansible', 'spring', 'data science', 'python', 'scikit.learn', 'virtual environments', 'bayesian inference', 'machine learning algorithms', 'teaching experience', 'machine learning', 'python', 'linux', 'java', 'scala', 'R+++', 'spark', 'java', 'python', 'scala', 'R+++', 'mapreduce', 'hive', 'spark', 'machine learning', 'python', 'data mining', 'hadoop', 'scala', 'spark', 'sql', 'R+++', 'python', 'sql', 'hadoop', 'algorithms', 'machine learning', 'data mining', 'python', 'machine learning', 'hadoop', 'aws', 'data mining', 'R+++', 'python', 'sql', 'hadoop', 'algorithms', 'machine learning', 'data mining', 'data analytics', 'healthcare industry', 'data science', 'sql or mysql', 'sas', 'statistical software', 'R+++', 'python', 'sas', 'spss', 'sql', 'R+++', 'python', 'sql', 'hadoop', 'algorithms', 'machine learning', 'data mining', 'machine learning', 'artificial intelligence', 'statistics', 'data analysis', 'unix', 'linux', 'statistical software', 'machine learning', 'data mining', 'python', 'hadoop', 'machine learning', 'python', 'data mining', 'R+++', 'phd preferred', 'machine learning', 'spark', 'java', 'python', 'scala', 'data science', 'from industry', 'machine learning algorithms', 'python', 'phenomenal written and oral communication', 'data manipulation', 'mining', 'modeling', 'machine learning', 'natural language processing', 'large data sets', 'hadoop', 'hands on team leadership', 'python', 'R+++', 'python', 'sql', 'hadoop', 'algorithms', 'machine learning', 'data mining', 'machine learning', 'data mining', 'hadoop', 'presto', 'sql server', 'mysql', 'machine learning', 'data mining', 'python', 'hadoop', 'scala', 'spark', 'sql', 'python', 'python scientific stack', 'numpy', 'pandas', 'scipy', 'etc.', 'basic statistics and data analysis', 'open source software', 'data ingestion and processing', 'machine learning', 'analytics experience', 'financial industry a plus', 'agile framework with atlassian', 'or similar', 'unclean', 'semi- structured', 'unstructured data', 'unsecured lending credit experience', 'python', 'basic statistics and data analysis', 'open source software', 'data ingestion and processing', 'machine learning', 'analytics experience', 'financial industry a plus', 'agile framework with atlassian', 'or similar', 'unclean', 'semi- structured', 'unstructured data', 'credit modeling', 'hadoop', 'spark', 'h20', 'data science', 'python', 'scikit.learn', 'virtual environments', 'bayesian inference', 'machine learning algorithms', 'prior teaching experience', 'not essential', 'tcp', 'ip', 'machine learning', 'data mining', 'hadoop', 'hive', 'cassandra', 'storm', 'or spark', 'information retrieval', 'regression', 'support vector machines', 'optimization', 'machine learning', 'artificial intelligence', 'statistics', 'data analysis', 'unix', 'linux', 'statistical software', 'data science', 'artificial intelligence', 'machine learning', 'masters or phd', 'cs', 'ai', 'machine learning', 'statistical analysis', 'open source machine', 'deep learning frameworks', 'pandas', 'python', 'numpy', 'jupyter', 'predictive modeling', 'hadoop', 'spark', 'mongodb', 'ruby', 'python', 'cyber security', 'scala', 'spark', 'machine learning', 'java', 'python', 'hadoop', 'scala', 'spark', 'machine learning', 'java', 'python', 'hadoop', 'python', 'c', 'c++', 'networking', 'security', 'apache spark', 'kafka', 'elasticsearch', 'mongodb', 'python', 'c', 'c++', 'networking', 'security', 'apache spark', 'kafka', 'elasticsearch', 'mongodb', 'data scientist', 'python', 'R+++', 'recommendation systems', 'machine learning', 'sql', 'python', 'pandas', 'jupyter', 'numpy', 'analytics', 'machine learning', 'text-mining', 'nlp', 'pathway', 'network analysis', 'modeling & simulations', 'java or c#', 'big data', 'cloud computing', 'pandas', 'python', 'numpy', 'jupyter', 'machine learning', 'python', 'R+++', 'sql', 'bi reporting', 'tableau', 'obiee', 'business objects', 'congos', 'micro strategy', 'sas', 'machine learning', 'python', 'R+++', 'sql', 'bi reporting', 'tableau', 'obiee', 'business objects', 'congos', 'micro strategy', 'sas', 'machine learning', 'deep learning', 'nlp', 'phd', 'ms in computer science', 'R+++', 'python', 'sql', 'hadoop', 'algorithms', 'machine learning', 'data mining', 'algorithm development', 'bi reporting tools', 'python', 'statistical modeling', 'online advertising', 'data pipeline architecture', 'statistical analysis & programming', 'redshift', 'sql', 'R+++', 'matlab', 'python', 'data mining', 'data analytics & visualization', 'python', 'matlab', 'R+++', 'python', 'aws', 'data science', 'data science', 'python', 'scikit.learn', 'virtual environments', 'bayesian inference', 'machine learning algorithms', 'prior teaching experience', 'not essential', 'json', 'machine learning', 'nlp', 'artificial intelligence', 'data transfer', 'chief scientist', 'solutions architect', 'big data', 'r or python', 'algorithm development', 'bi reporting tools', 'python', 'statistical modeling', 'online advertising', 'machine learning', 'data analysis', 'big data', 'data visualization', 'd3', 'predictive modeling', 'predictive analytics', 'data science', 'python', 'scikit.learn', 'virtual environments', 'bayesian inference', 'machine learning algorithms', 'teaching experience', 'R+++', 'python', 'sql', 'hadoop', 'algorithms', 'machine learning', 'data mining', 'machine learning', 'statistical modeling', 'data analytics', 'python', 'hadoop', 'aws', 'spark', 'mission critical systems', 'machine learning', 'phd', 'field analysis', 'python', 'R+++', 'matlab', 'sas', 'machine learning', 'python', 'linux', 'java', 'scala', 'R+++', 'spark', 'R+++', 'python', 'sql', 'hadoop', 'algorithms', 'machine learning', 'data mining', 'data science', 'from industry', 'machine learning algorithms', 'python', 'phenomenal written and oral communication', 'R+++', 'python', 'sql', 'aws', 'hadoop', 'predictive modeling', 'algorithm development', 'bi reporting tools', 'python', 'statistical modeling', 'online advertising', 'data science', 'python', 'scikit.learn', 'virtual environments', 'bayesian inference', 'machine learning algorithms', 'teaching experience', 'R+++', 'python', 'sql', 'hadoop', 'algorithms', 'machine learning', 'data mining', 'machine learning', 'java', 'scala', 'spark', 'python', 'data mining', 'R+++', 'bioinformatics', 'python', 'perl', 'java', 'R+++', 'python', 'sql', 'hadoop', 'algorithms', 'machine learning', 'data mining', 'machine learning', 'python', 'R+++', 'sql', 'bi reporting', 'tableau', 'obiee', 'business objects', 'congos', 'micro strategy', 'sas', 'R+++', 'matlab', 'cnn', 'rnn', 'python', 'scala', 'or java', 'gpu programming', 'big data', 'spark', 'data science', 'artificial intelligence', 'ai', 'machine learning', 'deep learning', 'robotics', 'security data', 'manufacturing data', 'healthcare data', 'machine learning', 'python', 'linux', 'java', 'scala', 'R+++', 'spark', 'statistics', 'spss', 'sas', 's-plus', 'R+++', 'analyzing large data sets', 'machine learning', 'clustering & classification', 'nlp', 'optimization algorithms', 'causal inference', 'deep learning', 'predictive modeling concepts', 'experiment design', 'computer vision', 'machine learning', 'python', 'scala', 'or java', 'hadoop', 'big data', 'R+++', 'matlab', 'c++', 'scala', 'spark', 'machine learning', 'java', 'python', 'hadoop', 'machine learning', 'hadoop', 'python', 'raw data analysis', 'data visualization', 'statistics', 'data manipulation', 'sql', 'R+++', 'tableau', 'data science', 'management', 'leadership', 'creativity', 'curiosity', 'machine learning', 'predictive modeling techniques', 'python', 'R+++', 'data visualization', 'game theory', 'machine learning', 'python', 'applied mathematics', 'R+++', 'software development', 'data scientist', 'deep learning', 'machine learning', 'computer vision', 'algorithm dev', 'python', 'java', 'hadoop', 'data scientist', 'python', 'hadoop', 'predictive analystics', 'novel machine learning', 'algorithms', 'ts', 'sci polygraph clearance', 'federal based engineering', 'c', 'c++', 'java', 'hadoop', 'python', 'numpy', 'pandas', 'matplotlib', 'd3js', 'tableau', 'machine learning', 'data mining', 'spss', 'sas', 'R+++', 'matlab', 'celect', 'datarobot', 'netica', 'data modeling', 'data warehousing', 'big data', 'hadoop', 'python', 'numpy', 'pandas', 'matplotlib', 'd3js', 'tableau', 'machine learning', 'data mining', 'data scientist', 'machine learning', 'R+++', 'numpy', 'spicy', 'pandas', 'hadoop mapreduce', 'apache spark', 'hive', 'probability theory and mathematical optimization', 'java', 'c++', 'unix', 'linux', 'machine learning', 'computer vision', 'caffe', 'knn', 'deep learning', 'data science', 'quantitative finance', 'systematic trading', 'proprietary trading', 'factor models', 'machine learning', 'machine learning', 'hadoop', 'python', 'raw data analysis', 'data visualization', 'statistics', 'data manipulation', 'sql', 'R+++', 'python', 'sql', 'data engineering', 'aws', 'machine learning', 'deep learning', 'nlp', 'phd', 'ms in computer science', 'data science', 'machine learning', 'sas', 'spss', 'R+++', 'python', 'statistical analysis', 'data science', 'machine learning', 'data mining', 'sql', 'python', 'scala', 'or java', 'distributed frameworks', 'machine learning', 'algorithms', 'financial statements', 'regression analysis', 'cluster analysis', 'R+++', 'sql', 'data mining', 'ph.d', 'data science', 'machine learning', 'data mining', 'sql', 'python', 'scala', 'or java', 'distributed frameworks', 'java', 'java springsource', 'c++', 'c#', 'data mining', 'machine learning', 'R+++', 'neo4j', 'rest-based apis', 'machine learning', 'algorithms', 'financial statements', 'regression analysis', 'cluster analysis', 'R+++', 'sql', 'data mining', 'ph.d', 'R+++', 'python', 'scikit learn', 'numpy', 'visual intelligence', 'aws', 'data science', 'clinical laboratory scientist', 'ascp', 'toxicology', 'data analysis', 'data manipulation', 'mining', 'modeling', 'machine learning', 'natural language processing', 'large data sets', 'hadoop', 'hands on team leadership', 'python', 'machine learning', 'graph analytics', 'statistical modeling', 'data analytics', 'python', 'R+++', 'hadoop', 'aws', 'food', 'microbiology', 'data compilation', 'data analysis', 'leadership', 'genome sequencing', 'proposals and reports', 'staffing', 'experiment execution', 'research scientist', 'machine learning', 'deep learning', 'nlp', 'artificial intelligence', 'neural networks', 'research scientist', 'machine learning', 'deep learning', 'nlp', 'artificial intelligence', 'neural networks', 'clustering', 'pattern recognition', 'nlp', 'mathematical', 'statistical modeling', 'natural language processing', 'neural networks', 'java', 'data mining', 'chief scientist', 'solutions architect', 'big data', 'r or python']


Here are all the results of filted preferred skills. 


```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt
wordcloud = WordCloud(max_words=50,background_color = 'white', width = 2800,height = 2400, max_font_size = 1000, font_path = "/Users/shishengjie/Desktop/cabin-sketch/CabinSketch-Regular.ttf").generate(','.join(skill))
plt.figure(figsize=(18,16))
plt.axis('off')
plt.imshow(wordcloud)
plt.show()
```


![png](output_43_0.png)


The above picture is data sciencetist wordcloud. There is a bug here for the wordcloud. Compared with the bar plot we generate for the preferred skill, we can see the skill **"R"** is one of the three preferred skills. But we cannot find **"R"** in the wordcloud picture. This problem also shown in the later SDE analysis. We guess the reason is that the algorithm of the wordcloud will igonre the single letter, such as **"R","C","C++"**.

Since we have the **"need for the position"** column in the dataset. We wonder the difference between the **"need for the position"** and **preferred skill**.


```python
# skill from need_for_position
ds_filtered_needForPosition = [word for word in tokenize(ds_needForPosition) if word not in stopwords.words('english') and word not in ['etc.','e.g.']] 
nltk.FreqDist(ds_filtered_needForPosition).plot(30)
```

    /usr/local/lib/python2.7/site-packages/ipykernel/__main__.py:2: UnicodeWarning: Unicode equal comparison failed to convert both arguments to Unicode - interpreting them as being unequal
      from ipykernel import kernelapp as app



![png](output_46_1.png)



```python
# experience required (not excluding empty entry)
ds_needForPosition_list = list(ds['need_for_position'])
ds_needForPosition_list_lower = list(ds['need_for_position'].str.lower()) # all lower case
len([i for i in ds_needForPosition_list_lower if 'experi' in i]) / float(len(ds_needForPosition_list_lower))
```




    0.8532110091743119



Almost 86% of the job posts required the applicants have the previous related experience in the industry. 

<center>** What's the situation of the software development engineer?**


```python
sde = pd.read_csv('Software_Engineer.csv', index_col=False)
del sde['Unnamed: 0']
sde_skill =",".join( sde['preferred_skill'] ).lower()
sde_filtered_skill = [word for word in tokenize(sde_skill) if word not in stopwords.words('english')] 
nltk.FreqDist(sde_filtered_skill).plot(30)
```


![png](output_50_0.png)



```python
# Two sets of words with intersection
sde_skill_words = pd.DataFrame(nltk.FreqDist(sde_filtered_skill).most_common(8) )
sde_skill_words.iloc[:,1] = sde_skill_words.iloc[:,1] / sde.shape[0] 
sde_barplot = sns.barplot( x = 0, y = 1,data = sde_skill_words, palette = 'Blues_d')
sde_barplot.set(xlabel = '', ylabel = 'percentage in SDE posts')
plt.show()
```


![png](output_51_0.png)



```python
wordcloud = WordCloud(max_words=50,background_color = 'white', width = 2800,height = 2400, max_font_size = 1000, font_path="/Users/shishengjie/Desktop/cabin-sketch/CabinSketch-Regular.ttf").generate(','.join(sde_filtered_skill))
plt.figure(figsize=(18,16))
plt.axis('off')
plt.imshow(wordcloud)
plt.show()
```


![png](output_52_0.png)


<h3><center>Degree<center></h3>



```python
# degree_requirement

degree_level = ['Master', ' MS','M.S','Ph.D','PhD', 'BS','Bachelor']
degree_field = ['statist','math','computer science','engineer','biolog', 'econ','physics','chemis', 'bioinformati', 'life science']
# 'cs' contained in 'analytics', 'physics', 
bachelor_total = 0; master_total = 0; phd_total = 0;
for i in ds['need_for_position']:
    master_total = master_total + sum( (x in i) for x in ['Master', 'MS','M.S'] ) # 'algorithms', 'systems','platforms'
    bachelor_total = bachelor_total + sum((x in i) for x in ['BS', 'Bachelor'])
    phd_total = phd_total + sum( (x in i) for x in ['PhD', 'Ph.D','phd','ph.d'])
print bachelor_total, master_total, phd_total

for k in degree_field:
    a = sum( k in x for x in ds_needForPosition_list_lower)
    print (k, a)
field = [['statistics', 'math','computer science', 'engineering', 'physics','life science','other'], [49, 37, 30, 22, 11, 10, 3]]
plt.figure(figsize=(8,6))
plt.pie(field[1], labels = field[0], autopct='%1.1f%%',colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','lightgrey','pink','darkorange'])
plt.axis('equal')
plt.title('Major Dist')
plt.show()
```

    0 0 46
    ('statist', 45)
    ('math', 32)
    ('computer science', 25)
    ('engineer', 21)
    ('biolog', 5)
    ('econ', 1)
    ('physics', 10)
    ('chemis', 2)
    ('bioinformati', 4)
    ('life science', 5)



![png](output_54_1.png)


The pie chart denotes that **Statistics, Math, Computer Science** are top three popular degrees that companies are welcome to hire no matter in the Data Science or SDE. 


```python
# degree_requirement

degree_level = ['Master', ' MS','M.S','Ph.D','PhD', 'BS','Bachelor']
degree_field = ['statist','math','computer science','engineer','biolog', 'econ','physics','chemis', 'bioinformati', 'life science']
# 'cs' contained in 'analytics', 'physics', 
bachelor_total1 = 0; master_total2 = 0; phd_total3 = 0; a = 0;

count = 0 
np.array([sum((k in i) for k in ['Master', 'MS','M.S']) for i in sde['need_for_position'] if not pd.isnull(i)]).sum()
```




    33



Also, there are 33 job posts specificly denoted that they like or preferred the master degree.  

In conclusion, according to the Cybercoder data, we get the most of employment information of Data Scientist and Software Develpment Engineer. Even though the current salary median of DS is lower than SDE, DS is a real potential job position for our statistic major students. Equited with some program languages like **"Python", "C"** and our professional statistical analysis experience, we believe that we can be really competitve in the job market. 
