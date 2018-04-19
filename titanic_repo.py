
# coding: utf-8

# # 选择分析泰坦尼克号项目，首先了解titanic-data.csv数据集，这些数据可以回答的问题包括。

# ## 1. 有哪些因素会让船上的人生还率更高？

# 数据集中的因素包括：
# 1	Survived	生存	0 =否，1 =是
# 2	Pclass	票类	1 = 1，2 = 2，3 = 3
# 3	Sex	性别	
# 4	Age	年龄	
# 5	SibSp	泰坦尼克号上的兄弟姐妹/配偶	
# 6	Parch	泰坦尼克号上的父母/孩子的数量	
# 7	Ticket	票号	
# 8	Fare	乘客票价	
# 9	Cabin	客舱号码	
# 10	Embarked	登船港口	C =瑟堡，Q =皇后镇，S =南安普敦
# 
# ### 我分析可能的因素为1、性别  2、年龄   3、票类/票价   4.孩子有父母和女性有配偶或兄弟生还率将增大
# 

# In[1]:


# 导入相关包数据
get_ipython().magic(u'pylab inline')
import numpy as py
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)


# In[2]:


# 导入数据集
data = pd.read_csv('titanic-data.csv')
data['Fare_1'] = ''
data['Age_1'] = ''


# In[3]:


# 显示前三条数据
data.head(50)


# In[4]:


# 总人数
pepole_sum = data.count()['Name']
pepole_sum 


# In[5]:


def fare_cmp(para):
    if para > 200:
        return 'A'
    elif para > 150:
        return 'B'
    elif para > 100:
        return 'C'
    elif para >50:
        return 'D'
    else:
        return 'E'


# In[250]:


def age_cmp(para):
    if para < 12:
        return 'A'
    elif para < 18:
        return 'B'
    else:
        return 'C'


# In[251]:


data['Fare_1'] = data['Fare'].map(fare_cmp)


# In[252]:


data['Age_1']=data['Age'].map(age_cmp)


# In[254]:


def ratio(data_cache, key_col, live):
    # 返回key_col分类的live的比率，live为1为生存率，0为死,cata为具体类别
    result = []
    group_cache = data_cache.groupby([key_col, 'Survived'])
    cata_key = data_cache.groupby([key_col, 'Survived']).size().index.levels[0].values
    for i in cata_key:
        print '分类基数' + str(i) + ':' + str(group_cache.size()[i].sum())
        result.append(float(group_cache.size()[i, live]) / group_cache.size()[i].sum())
    print pd.Series(result, index=cata_key)
    return pd.Series(result, index=cata_key).plot(kind='bar')


# In[255]:


# 分类中SEX类别生存率
ratio(data, 'Sex', 1)


# ## 分析
# 从上图中可以看出女性的生还率是男性的3.5倍还要多，所以猜测女性有更高的生还机会，猜测主要由于英国男人比较绅士，把生的机会让给女性。

# In[256]:


# 分类中Fare类别生存率
ratio(data, 'Fare_1', 1)


# ## 分析
# 此为票价分析：
# 'A'>$200、'B'> $150、'C'>$100、'D'>$50、'E'<$50
# 从图中看到，购买高价位票的人员，生还比率高，购买票价大于$50的人员和购买更贵的票的人员比购买低于$50美元的人员的生还率高一倍。所以推断富有的人员生还率更高。
# 

# In[219]:


# 分类中Pclass类别生存率
ratio(data, 'Pclass', 1) 


# ## 分析
# 从数据看一二等仓的生还率较高，比三等仓分别高出2.6倍和2倍，所以推断富有的人生还率更高。

# In[220]:


# 分类中Embarked类别生存率
ratio(data, 'Embarked', 1) 


# ## 分析
# 不同的港口，生还率相近

# In[257]:


# 分类中Age类别生存率,其中删除了Age==NaN的行
data_mu_age = data.dropna(subset = ['Age'])
ratio(data_mu_age, 'Age_1', 1) 


# ## 分析
# A代表12岁以下孩子的生还率，B代表18岁以下，C代表18岁以上的人，12岁以下孩子的生还率是18岁以上人员的1.5倍，12到18岁的人员生还率又有减少。

# In[258]:


def ratio_three(data_cache, key_1, key_2, key_3):
    # 返回key_col分类的live的比率，live为1为生存率，0为死,cata为具体类别
    result = []
    group_cache = data_cache.groupby([key_1, key_2, key_3, 'Survived'])
    cata_key = group_cache.size().index.values
    cata_key_1 = []
    for w in cata_key:
        if w[3]!=0L:
            cata_key_1.append(w)
    for i in cata_key_1:
        (a, b, c, d) = i
        print '分类基数' + str([a, b, c]) + ':' + str(group_cache.size()[a, b, c].sum())
        result.append(float(group_cache.size()[a, b, c, d]) / group_cache.size()[a, b, c].sum())
    return pd.Series(result, index=cata_key_1).plot(kind='bar')


# In[249]:


ratio_three(data_mu_age, 'Pclass', 'Sex', 'Age_1')


# ## 结论：删除没有年龄的人员后，从图中推论富有的女性和孩子生还率高。由于其中的限制条件和位置因素很多，例如随行人员的影响，所以此次只是从部分数据中推论如此，此次的数据表现的只是相关性，并无因果关系，随着分析的完善，可能会有更多的发现。
