#!/usr/bin/env python
# coding: utf-8

# # <span style="color:blue">Some useful functions and code</span>

# **M. Emile F. Apol**, *Life Science & Technology, Hanze University of Applied Sciences, Groningen*

# Version: 15
# 
# Latest update: 2024-01-22

# Overview of functions and code:
# 
# *Viewing distribution and checking for normality:*
# 
# 1. **DS_Q_Q_Plot**:	Make a Q-Q plot for a normal distribution with 95% CI lines
# 2. **DS_Q_Q_Hist**:	Make a histogram with fitted normal distribution
# 3. **DS_AndersonDarling_test_normal**:	Anderson-Darling test for normality of data
# 
# *Tests for means (assuming normal distribution, sigma2 known):*
# 
# 4. **DS_1sample_ztest_means**:	1-sample z-test for means mu (1- and 2-sided, known standard deviation)
# 5. **DS_2sample_ztest_means**:	2-sample z-test for means mu (1- and 2-sided, known standard deviations)
# 6. **DS_paired_ztest_means**:	Paired z-test for means mu (1- and 2-sided, known standard deviation of differences)
# 
# *Tests for means (assuming normal distribution, sigma2 unknown):*
# 
# 7. **DS_1sample_ttest_means**:	1-sample t-test for means mu (1- and 2-sided, unknown standard deviation)
# 8. **DS_2sample_ttest_means**:	2-sample (Welch's) t-test for means mu (1- and 2-sided, unknown standard deviations)
# 9. **DS_paired_ttest_means**:	Paired t-test for means mu (1- and 2-sided, unknown standard deviation of differences)
# 
# *Tests for variances (assuming normal distribution):*
# 
# 10. **DS_1sample_chi2test_vars**:	1-sample chi2-test for 1 variance sigma2 (1- and 2-sided)
# 11. **DS_2sample_Ftest_vars**:	F-test for 2 variances sigma2 (1- and 2-sided), assuming normal distribution
# 12. **DS_2sample_Levenetest_vars**:	2-sample Levene-Brown-Forsythe-test for 2 variances sigma2 (1- and 2-sided)
# 13. **DS_paired_PitmanMorgan_test_vars**:	Paired Pitman-Morgan test for 2 variances (1- and 2-sided)
# 
# *Tests for medians (not assuming a normal distribution):*
# 
# 14. **DS_1sample_Wilcoxon_test_medians**:	1-sample Wilcoxon signed-rank test for 1 median (1- and 2-sided)
# 15. **DS_2sample_MannWhitney_test_medians**:	2-sample Mann-Whitney U-test for 2 medians (1- and 2-sided)
# 16. **DS_paired_Wilcoxon_test_medians**:	Paired Wilcoxon test for 2 medians
# 
# *Tests for proportions (assuming a Bernoulli or Binomial distribution):*
# 
# 17. **DS_1sample_ztest_props**:	1-sample z-test for 1 proportion p (1- and 2-sided)
# 18. **DS_2sample_ztest_props**:	2-sample z-test for 2 proportions p (1- and 2-sided)
# 19. **DS_paired_McNemar_chi2_test_props**:	paired McNemar chi2-test for 2 proportions (1- and 2-sided)
# 
# *Tests for rates and counts (assuming a Poisson distribution):*
# 
# 20. **DS_1sample_ztest_counts**:	1-sample z-test for Poisson counts lambda (1- and 2-sided)
# 21. **DS_2sample_ztest_counts**:	2-sample z-test for Poisson counts lambda (1- and 2-sided)
# 
# *Analysis of calibration curves (linear regression):*
# 
# 22. **DS_CalibrationAnalysis**:	Analysis of calibration models: fitting, model selection, plotting, interpolation
# 23. **DS_OLS_AIC**:	Calculate the AIC and AIC.c-values from a statsmodels.api OLS fit
# 24. **DS_OLS_predict_with_CI**:	Calculate the predicted y-values with CI from a statsmodels.api OLS fit
# 
# *Other extra functions:*
# 
# 25. **DS_xtab**:	Make a 1d or 2d contingency table from 1 or 2 arrays (without Pandas)
# 26. **DS_Beta_NBinom.pmf**, **DS_Beta_NBinom.cdf**, **DS_Beta_NBinom.ppf**, **DS_Beta_NBinom.rvs**: The Beta-Negative Binomial distribution (not yet in scipy.stats)
# 27. **DS_Mu_r**:	Calculate the central sample moment m.r  from a 1d array
# 28. **fsolve**:		Numerically solving one ML equation (1 parameter)
# 29. **fsolve**:		Numerically solving two ML equations (2 parameters)

# ========================================================================================

# To import this file into your notebook, you could use the following:

# In[1]:


"""
import sys
import os
sys.path.append(os.path.abspath("/your_dir_structure_where_the_Summary.py_function_is"))

from Summary_and_functions_clean_v12 import *
"""


# ## <span style="color:blue">*Viewing distribution and checking for normality:*</span>

# ## 1.  Make a Q-Q plot for a normal distribution with 95% CI lines

# In[2]:


def DS_Q_Q_Plot(y, est = 'robust', title = 'Q-Q plot', **kwargs):
    """
    *
    Function DS_Q_Q_Plot(y, est = 'robust', **kwargs)
    
       This function makes a normal quantile-quantile plot (Q-Q-plot), also known
       as a probability plot, to visually check whether data follow a normal distribution.
    
    Requires:            numpy, scipy.stats.iqr, scipy.stats.norm, matplotlib.pyplot
    
    Arguments:
      y                  data array
      est                Estimation method for normal parameters mu and sigma:
                         either 'robust' (default), or 'ML' (Maximum Likelihood),
                         or 'preset' (given values)
      N.B. If est='preset' than the *optional* parameters mu, sigma must be provided:
      mu                 preset value of mu
      sigma              preset value of sigma
      title              Title for the graph (default: 'Q-Q plot')
      
    Returns:
      Estimated mu, sigma, n, and expected number of datapoints outside CI in Q-Q-plot.
      Q-Q-plot
      
    Author:            M.E.F. Apol
    Date:              2020-01-06, revision 2022-08-30, 2023-12-19
    """
    
    import numpy as np
    from scipy.stats import iqr # iqr is the Interquartile Range function
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    
    # First, get the optional arguments mu and sigma:
    mu_0 = kwargs.get('mu', None)
    sigma_0 = kwargs.get('sigma', None)
    
    n = len(y)
    
    # Calculate order statistic:
    y_os = np.sort(y)
  
    # Estimates of mu and sigma:
    # ML estimates:
    mu_ML = np.mean(y)
    sigma2_ML = np.var(y)
    sigma_ML = np.std(y) # biased estimate
    s2 = np.var(y, ddof=1)
    s = np.std(y, ddof=1) # unbiased estimate
    # Robust estimates:
    mu_R = np.median(y)
    sigma_R = iqr(y)/1.349

    # Assign values of mu and sigma for z-transform:
    if est == 'ML':
        mu, sigma = mu_ML, s
    elif est == 'robust':
        mu, sigma = mu_R, sigma_R
    elif est == 'preset':
        mu, sigma = mu_0, sigma_0
    else:
        print('Wrong estimation method chosen!')
        return()
        
    print('Estimation method: ' + est)
    print('n = {:d}, mu = {:.4g}, sigma = {:.4g}'.format(n, mu,sigma))
    
    # Expected number of deviations (95% confidence level):
    n_dev = np.round(0.05*n)
    
    print('Expected number of data outside CI: {:.0f}'.format(n_dev))
         
    # Perform z-transform: sample quantiles z.i
    z_i = (y_os - mu)/sigma

    # Calculate cumulative probabilities p.i:
    i = np.array(range(n)) + 1
    p_i = (i - 0.5)/n

    # Calculate theoretical quantiles z.(i):
    z_th = norm.ppf(p_i, 0, 1)

    # Calculate SE or theoretical quantiles:
    SE_z_th = (1/norm.pdf(z_th, 0, 1)) * np.sqrt((p_i * (1 - p_i)) / n)

    # Calculate 95% CI of diagonal line:
    CI_upper = z_th + 1.96 * SE_z_th
    CI_lower = z_th - 1.96 * SE_z_th

    # Make Q-Q plot:
    plt.plot(z_th, z_i, 'o', color='k', label='experimental data')
    plt.plot(z_th, z_th, '--', color='r', label='normal line')
    plt.plot(z_th, CI_upper, '--', color='b', label='95% CI')
    plt.plot(z_th, CI_lower, '--', color='b')
    plt.xlabel('Theoretical quantiles, $z_{(i)}$')
    plt.ylabel('Sample quantiles, $z_i$')
    plt.title(title + ' (' + est + ')')
    plt.legend(loc='best')
    plt.show()
    pass;


# ## 2.  Make a histogram with fitted normal distribution

# In[5]:


def DS_Q_Q_Hist(y, est='robust', title = 'Histogram with corresponding normal distribution', 
                xlabel = 'Values, $y$', ylabel = 'Probability $f(y)$', bins='auto', align='mid', **kwargs):
    """
    *
    Function DS_Q_Q_Hist(y, est='robust', **kwargs)
    
       This function makes a histogram of the data and superimposes a fitted normal
       distribution.
       
    Requires:            - 
    
    Arguments:
      y                  data array
      est                Estimation method for normal parameters mu and sigma:
                         either 'robust' (default), or 'ML' (Maximum Likelihood),
                         or 'MM' (Method of Moments) or 'preset' (given values)
      N.B. If est='preset' than the optional parameters mu, sigma MUST be provided:
      mu                 preset value of mu
      sigma              preset value of sigma
      title              Title of the graph (default: 'Histogram with corresponding normal distribution')
      xlabel             x-label of the graph (default: 'Values, $y$')
      ylabel             y-label of the graph (default: 'Probability $f(y)$')
      bins               bins used for histogram (default: 'auto'); can be np.array of bins
      align              align the histogram bins (default: 'mid'); for already binned data the option 'left' might
                         give better results
    
    Returns:
      Estimations of mu and sigma
      Histogram of data with estimated normal distribution superimposed
      
    Author:            M.E.F. Apol
    Date:              2020-01-06, adaptions 2023-12-19
    """
    
    import numpy as np
    from scipy.stats import iqr # iqr is the Interquartile Range function
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    
    # First, get the optional arguments mu and sigma:
    mu_0 = kwargs.get('mu', None)
    sigma_0 = kwargs.get('sigma', None)
    
    n = len(y)
    
    # Estimates of mu and sigma:
    # ML estimates:
    mu_ML = np.mean(y)
    sigma2_ML = np.var(y) # biased estimate
    sigma_ML = np.std(y) 
    s2 = np.var(y, ddof=1) # unbiased estimate
    s = np.std(y, ddof=1) 
    # Robust estimates:
    mu_R = np.median(y)
    sigma_R = iqr(y)/1.349

    # Assign values of mu and sigma for z-transform:
    if est == 'ML':
        mu, sigma = mu_ML, s    
    elif est == 'MM':
        mu, sigma = mu_ML, sigma_ML
    elif est == 'robust':
        mu, sigma = mu_R, sigma_R
    elif est == 'preset':
        mu, sigma = mu_0, sigma_0
    else:
        print('Wrong estimation method chosen!')
        return()
    print('Estimation method: ' + est)
    print('mu = {:.4g}, sigma = {:.4g}'.format(mu,sigma))
        
    # Calculate the CLT normal distribution:
    if bins != 'auto':
        y_min = np.min([np.min(y), np.min(bins)])
        y_max = np.max([np.max(y), np.max(bins)])
    else:
        y_min = np.min(y)
        y_max = np.max(y)
    x = np.linspace(y_min, y_max, 501)
    rv = np.array([norm.pdf(xi, loc = mu, scale = sigma) for xi in x])
    
    # Make a histogram with corresponding normal distribution:
    plt.hist(x=y, density=True, bins=bins, align=align,
             color='darkgrey',alpha=1, rwidth=1, label='experimental')
    plt.plot(x, rv, 'r', label='normal approximation')
    plt.grid(axis='y', alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title + ' (' + est + ')')
    plt.legend(loc='best')
    plt.show()
    pass;


# ## 3. Anderson-Darling-test for normality

# In[8]:


def DS_AndersonDarling_test_normal(y, alpha=0.05):
    """
    *
    Function DS_AndersonDarling_test_normal(y, alpha)
    
       This function tests whether the data y follow a normal distribution (Null Hypothesis Significance Test).
    
    Requires:            scipy.stats.anderson
    
    References:          * Th. Anderson & D. Darling (1952) - "Asymptotic Theory of 
                         Certain "Goodness of Fit" Criteria Based on Stochastic Processes".
                         Ann. Math. Statist. 23, 193-212. DOI: 10.1214/aoms/1177729437
                         * R.B. D'Agostino (1986). "Tests for the Normal Distribution"
                         In: R.B. D'Augostino & M.A. Stephens - "Goodness-of-fit
                         techniques", Marcel Dekker.
    
    Arguments:
      y                  data array
      alpha              significance level of the critical value (default: alpha = 0.05)
      
    Usage:               DS_AndersonDarling_test_normal(y, alpha=alpha)
      
      
    Returns:             AD, AD_star, p_value [ + print interpretable output to stdout ]
    where
      AD                 (Large-sample) Anderson-Darling statistic
      AD_star            Small-sample Anderson-Darling statistic
      p_value            p-value of AD-test
      
    Author:            M.E.F. Apol
    Date:              2023-12-05
    """
    
    from scipy.stats import anderson
    
    AD = anderson(y, dist='norm').statistic
    n = len(y)
    AD_star = AD*(1 + 0.75/n + 2.25/n**2)
    
    # p-values based on D'Augostino & Stephens (1986):
    if(AD_star <= 0.2): # Eq. (1)
        p_value = 1 - np.exp(-13.436 + 101.14*AD_star - 223.73*AD_star**2)
    elif((AD_star > 0.2) & (AD_star <= 0.34)): # Eq. (2)
        p_value = 1 - np.exp(-8.318 + 42.796*AD_star - 59.938*AD_star**2)
    elif((AD_star > 0.34) & (AD_star < 0.6)): # Eq. (3)
        p_value = np.exp(0.9177 - 4.279*AD_star - 1.38*AD_star**2)
    elif(AD_star >= 0.6): # Eq. (4)
        p_value = np.exp(1.2937 - 5.709*AD_star + 0.0186*AD_star**2)
        
    # Critical AD* values, based on D'Augostino & Stephens (1986):
    # Inverting these relations, we get
    # Invert (1) if alpha > 0.884
    # Invert (2) if 0.50 < alpha < 0.884
    # Invert (3) if 0.1182 < alpha < 0.50
    # Invert (4) if alpha < 0.1182
    
    if(alpha >= 0.884): # Eq. (1a)
        AD_crit = (-101.14+np.sqrt(101.14**2-4*-223.73*(-13.436-np.log(1-alpha))))/(2* -223.73)
    elif((alpha < 0.884) & (alpha >= 0.50)): # Eq. (2a)
        AD_crit = (-42.796+np.sqrt(42.796**2-4* -59.938*(-8.318-np.log(1-alpha))))/(2* -59.938)
    elif((alpha < 0.50) & (alpha >= 0.1182)): # Eq. (3a)
        AD_crit = (4.279-np.sqrt(4.279**2-4* -1.38*(0.9177-np.log(alpha))))/(2* -1.38)
    elif(alpha < 0.1182): # Eq. (4a)
        AD_crit = (5.709-np.sqrt(5.709**2-4*0.0186*(1.2937-np.log(alpha))))/(2*0.0186)
    
    # Additional statistics:
    y_av = np.mean(y)
    s = np.std(y, ddof=1)
    
    print(80*'-')
    print('Anderson-Darling-test for normality of data:')
    print('     assuming Normal(mu | sigma2) data for dataset')
    print('y.av = {:.3g}, s = {:.3g}, n = {:d}, alpha = {:.3g}'.format(y_av, s, n, alpha))
    print('H0: data follows normal distribution')
    print('H1: data does not follow normal distribution')
    print('AD = {:.3g}, AD* = {:.3g}, p-value = {:.3g}, AD*.crit = {:.3g}'.format(AD, AD_star, p_value, AD_crit))
    print(80*'-')
    
    return(AD, AD_star, p_value);


# ## <span style="color:blue">*Tests for means (assuming normal distribution, $\sigma^2$ known):*</span>

# ## 4. 1-sample $z$-test for means (1- and 2-sided, with known standard deviation)

# In[12]:


def DS_1sample_ztest_means(y, sigma, popmean=0, alternative='two-sided', alpha=0.05):
    """
    *
    Function DS_1sample_ztest_means(y, sigma, popmean=0, alternative='two-sided', alpha=0.05)
     
        This function performs a 1-sample z-test (Null Hypothesis Significance Test) 
        in the spirit of R, testing 1 average with *known* standard deviation.
        The function also evaluates the effect size (Cohen's d).
     
    Author:            M.E.F. Apol
    Date:              2022-01-27, rev. 2022_08_26
    Validation:
    
    Requires:          -
    
    Usage:             DS_1sample_ztest_means(y, sigma = sigma, popmean = mu*, 
                            alternative=['two-sided']/'less'/'greater', alpha = 0.05)
     
                         alternative = 'two-sided' [default]  H1: mu != mu*
                                       'less'                 H1: mu < mu*
                                       'greater'              H1: mu > mu*
                         sigma: *known* standard deviation of dataset
                         alpha: significance level of test [default: 0.05]
     
    Return:            z, p-value, z.crit.L, z.crit.R  [ + print interpretable output to stdout ]
                       where z.crit.L and z.crit.R are the lower and upper critical values, 
                       z is the test statistic and p-value is the p-value of the test.
     
    """
    
    import numpy as np
    from scipy.stats import norm
    
    n = len(y)
    y_av = np.mean(y)
    z = (y_av - popmean)/(sigma / np.sqrt(n))
    
    print(80*'-')
    print('1-sample z-test for 1 mean:')
    print('     assuming Normal(mu | sigma2) data for dataset')
    print('y.av = {:.3g}, mu* = {:.3g}, sigma = {:.3g}, n = {:d}, alpha = {:.3g}'.format(y_av, popmean, sigma, n, alpha))
    print('H0: mu  = mu*')
    
    if alternative == 'two-sided':
        print('H1: mu != mu*')
        p_value = 2 * norm.cdf(-np.abs(z), 0, 1)
        z_crit_L = norm.ppf(alpha/2, 0, 1)
        z_crit_R = norm.ppf(1-alpha/2, 0, 1)
    elif alternative == 'less':
        print('H1: mu  < mu*')
        p_value = norm.cdf(z)
        z_crit_L = norm.ppf(alpha, 0, 1)
        z_crit_R = float('inf')
    elif alternative == 'greater':
        print('H1: mu  > mu*')
        p_value = 1 - norm.cdf(z, 0, 1)
        # better precision, use the survival function:
        p_value = norm.sf(z, 0, 1)
        z_crit_L = float('-inf')
        z_crit_R = norm.ppf(1-alpha, 0, 1)
    else:
        print('Wrong alternative hypothesis chosen!')
        print(80*'-' + '\n')
        z, p_value, z_crit_L, z_crit_R = np.nan, np.nan, np.nan, np.nan
        return(z, p_value, z_crit_L, z_crit_R)
    
    # Effect size (Cohen's d.s):
    d_s = z * np.sqrt(1/n)
    print('z = {:.4g}, p-value = {:.4g}, z.crit.L = {:.4g}, z.crit.R = {:.4g}'.format(z, p_value, z_crit_L, z_crit_R))
    print(80*'.')
    print('Effect size: Cohen\'s d.s = {:.3g}; benchmarks |d.s|: 0.2 = small, 0.5 = medium, 0.8 = large'.format(d_s))
    print(80*'-' + '\n')
    return(z, p_value, z_crit_L, z_crit_R)


# ## 5.  2-sample $z$-test for means (1- and 2-sided, with known standard deviations)

# In[15]:


def DS_2sample_ztest_means(y1, y2, sigma1, sigma2, alternative ='two-sided', alpha=0.05):
    """
    *
    Function DS_2sample_ztest_means(y1, y2, sigma1, sigma2, alternative ='two-sided', alpha=0.05)
     
        This function performs a 2-sample z-test (Null Hypothesis Significance Test)
        in the spirit of R, testing 2 averages with *known* standard deviations.
        The function also evaluates the effect size (Cohen's d).
    
    Requires:          -
    
    Usage:             DS_2sample_ztest_means(y1, y2, sigma1, sigma2, 
                              alternative=['two-sided']/'less'/'greater', alpha = 0.05)
     
                         alternative = 'two-sided' [default]  H1: mu1 != mu_2
                                       'less'                 H1: mu1 < mu2
                                       'greater'              H1: mu1 > mu2
                         sigma1, sigma2: *known* standard deviations of datasets 1 and 2
                         alpha: significance level of test [default: 0.05]
     
    Return:            z, p-value, z.crit.L, z.crit.R  [ + print interpretable output to stdout ]
                       where z.crit.L and z.crit.R are the lower and upper critical values, 
                       z is the test statistic and p-value is the p-value of the test. 
     
    Author:            M.E.F. Apol
    Date:              2022-01-27, rev. 2022_08_26
    Validation:
    """
    
    import numpy as np
    from scipy.stats import norm
    
    n_1 = len(y1) ; n_2 = len(y2)
    y_av_1 = np.mean(y1) ; y_av_2 = np.mean(y2)
    z = (y_av_1 - y_av_2)/np.sqrt( sigma1**2/n_1 + sigma2**2/n_2 )
    
    print(80*'-')
    print('2-sample z-test for 2 means:')
    print('     assuming Normal(mu.1 | sigma2.1) data for dataset 1')
    print('     assuming Normal(mu.2 | sigma2.2) data for dataset 2')
    print('y.av.1 = {:.3g}, y.av.2 = {:.3g}, sigma.1 = {:.3g}, sigma.2 = {:.3g}, n.1 = {:d}, n.2 = {:d}, alpha = {:.3g}'.format(y_av_1, y_av_2, sigma1, sigma2, n_1, n_2, alpha))
    print('H0: mu.1  = mu.2')
    
    if alternative == 'two-sided':
        print('H1: mu.1 != mu.2')
        p_value = 2 * norm.cdf(-np.abs(z), 0, 1)
        z_crit_L = norm.ppf(alpha/2, 0, 1)
        z_crit_R = norm.ppf(1-alpha/2, 0, 1)
    elif alternative == 'less':
        print('H1: mu.1  < mu.2')
        p_value = norm.cdf(z)
        z_crit_L = norm.ppf(alpha, 0, 1)
        z_crit_R = float('inf')
    elif alternative == 'greater':
        print('H1: mu.1  > mu.2')
        p_value = 1 - norm.cdf(z, 0, 1)
        # better precision, use the survival function:
        p_value = norm.sf(z, 0, 1)
        z_crit_L = float('-inf')
        z_crit_R = norm.ppf(1-alpha, 0, 1)
    else:
        print('Wrong alternative hypothesis chosen!')
        print(80*'-' + '\n')
        z, p_value, z_crit_L, z_crit_R = np.nan, np.nan, np.nan, np.nan
        return(z, p_value, z_crit_L, z_crit_R)
    
    # Effect size (Cohen's d.av):
    sigma_pooled = np.sqrt((n_1*sigma1**2 + n_2*sigma2**2)/(n_1 + n_2))
    d_av = (y_av_1 - y_av_2)/sigma_pooled
    print('z = {:.4g}, p-value = {:.4g}, z.crit.L = {:.4g}, z.crit.R = {:.4g}'.format(z, p_value, z_crit_L, z_crit_R))
    print(80*'.')
    print('Effect size: Cohen\'s d.av = {:.3g}; benchmarks |d.av|: 0.2 = small, 0.5 = medium, 0.8 = large'.format(d_av))
    print(80*'-' + '\n')
    return(z, p_value, z_crit_L, z_crit_R)


# ## 6. Paired $z$-test for means (1- and 2-sided, with known standard deviation of the differences)

# In[18]:


def DS_paired_ztest_means(y1, y2, sigma_d, alternative ='two-sided', alpha=0.05):
    """
    *
    Function DS_paired_ztest_means(y1, y2, sigma_d, alternative ='two-sided', alpha=0.05)
     
       This function performs a paired z-test (Null Hypothesis Significance Test)
       in the spirit of R, testing the average difference between two sets of *paired* data 
       with *known* standard deviation of the difference, sigma_d.
    
    Requires:          -
    
    Usage:             DS_paired_ztest_means(y1, y2, sigma_d = sigma, 
                           alternative=['two-sided']/'less'/'greater', alpha = 0.05)
     
                         alternative = 'two-sided' [default]  H1: mu != mu*
                                       'less'                 H1: mu < mu*
                                       'greater'              H1: mu > mu*
                         sigma_d: known standard deviation of the differences
                         alpha:   significance level of test [default: 0.05]
     
    Return:            z, p-value, z.crit.L, z.crit.R  [ + print interpretable output to stdout ]
                       where z.crit.L and z.crit.R are the lower and upper critical values, 
                       z is the test statistic and p-value is the p-value of the test. 
     
    Author:            M.E.F. Apol
    Date:              2022-01-27, rev. 2022_08_26
    Validation:
    """
    
    import numpy as np
    from scipy.stats import norm
    
    # Error checking:
    if len(y1) != len(y2):
        print('Error: Datasets of unequal length...')
        return
    
    y_av_1 = np.mean(y1); y_av_2 = np.mean(y2)
    d_av = y_av_1 - y_av_2
    n_d = len(y1)
    z = (d_av)/(sigma_d / np.sqrt(n_d))
    
    print(80*'-')
    print('Paired z-test for 2 means:')
    print('     assuming Normal(mu.d |sigma2.d) data for difference between datasets 1 and 2')
    print('y.av.1 = {:.3g}, y.av.2 = {:.3g}, sigma.d = {:.3g}, n.1 = {:d}, n.2 = {:d}, alpha = {:.3g}'.format(y_av_1, y_av_2, sigma_d, n_d, n_d, alpha))
    print('H0: mu.1  = mu.2')
    
    if alternative == 'two-sided':
        print('H1: mu.1 != mu.2')
        p_value = 2 * norm.cdf(-np.abs(z), 0, 1)
        z_crit_L = norm.ppf(alpha/2, 0, 1)
        z_crit_R = norm.ppf(1-alpha/2, 0, 1)
    elif alternative == 'less':
        print('H1: mu.1  < mu.2')
        p_value = norm.cdf(z)
        z_crit_L = norm.ppf(alpha, 0, 1)
        z_crit_R = float('inf')
    elif alternative == 'greater':
        print('H1: mu.1  > mu.2')
        p_value = 1 - norm.cdf(z, 0, 1)
        # better precision, use the survival function:
        p_value = norm.sf(z, 0, 1)
        z_crit_L = float('-inf')
        z_crit_R = norm.ppf(1-alpha, 0, 1)
    else:
        print('Wrong alternative hypothesis chosen!')
        print(80*'-' + '\n')
        z, p_value, z_crit_L, z_crit_R = np.nan, np.nan, np.nan, np.nan
        return(z, p_value, z_crit_L, z_crit_R)
    print('z = {:.4g}, p-value = {:.4g}, z.crit.L = {:.4g}, z.crit.R = {:.4g}'.format(z, p_value, z_crit_L, z_crit_R))
    print(80*'-' + '\n')
    return(z, p_value, z_crit_L, z_crit_R)


# ## <span style="color:blue">*Tests for means (assuming normal distribution, $\sigma^2$ unknown):*</span>

# ## 7. 1-sample $t$-test (1- and 2-sided)

# In[22]:


def DS_1sample_ttest_means(y, popmean=0, alternative = 'two-sided', alpha=0.05):
    """
    *
    Function DS_1sample_ttest_means(y, popmean=0, alternative = 'two-sided', alpha=0.05)
    
       This function performs a 1-sample t-test (Null Hypothesis Significance Test) 
       in the spirit of R, testing 1 average with *unknown* standard deviation.
       The function also evaluates the effect size (Cohen's d).
       
    Requires:          -
     
    Usage:             DS_1sample_ttest_means(y, popmean = mu*, 
                            alternative=['two-sided']/'less'/'greater', 
                            alpha = 0.05)
     
                         alternative = 'two-sided' [default]  H1: mu != mu*
                                       'less'                 H1: mu < mu*
                                       'greater'              H1: mu > mu*
                         alpha:   significance level of test [default: 0.05]
     
    Return:            t, p-value, t.crit.L, t.crit.R  [ + print interpretable output to stdout ]
                       where t.crit.L and t.crit.R are the lower and upper critical values, 
                       t is the test statistic and p-value is the p-value of the test.
                       
    Author:            M.E.F. Apol
    Date:              2022-01-27, rev. 2022_08_26
    Validation:
    """
    
    from scipy.stats import ttest_1samp
    from scipy.stats import t as t_distr
    import numpy as np

    t, p_samp = ttest_1samp(y, popmean)
    print(80*'-')
    print('1-sample t-test for 1 mean:')
    print('     assuming Normal(mu, sigma2) data for dataset')
    y_av = np.mean(y)
    n = len(y)
    df = n - 1
    s2 = np.var(y, ddof=1)
    print('y.av = {:.3g}, mu* = {:.3g}, s2 = {:.3g}, n = {:d}, alpha = {:.3g}'.format(y_av, popmean, s2, n, alpha))
    print('H0: mu  = mu*')
    if alternative == 'two-sided':
        print('H1: mu != mu*')
        p_value = p_samp
        t_crit_L = t_distr.ppf(alpha/2, df)
        t_crit_R = t_distr.ppf(1-alpha/2, df)
    elif alternative == 'less':
        print('H1: mu  < mu*')
        if t <= 0:
            p_value = p_samp/2
        else:
            p_value = 1 - p_samp/2
        t_crit_L = t_distr.ppf(alpha, df)
        t_crit_R = float('inf')
    elif alternative == 'greater':
        print('H1: mu  > mu*')
        if t >= 0:
            p_value = p_samp/2
        else:
            p_value = 1 - p_samp/2
        t_crit_L = float('-inf')
        t_crit_R = t_distr.ppf(1-alpha, df)
    else:
        print('Wrong alternative hypothesis chosen!')
        print(80*'-' + '\n')
        t, p_value, t_crit_L, t_crit_R = np.nan, np.nan, np.nan, np.nan
        return(t, p_value, t_crit_L, t_crit_R)
    # Effect size:
    d_s = t * np.sqrt(1/n)
    print('t = {:.4g}, p-value = {:.4g}, t.crit.L = {:.4g}, t.crit.R = {:.4g}, df = {:.4g}'.format(t, p_value, t_crit_L, t_crit_R, df))
    print(80*'.')
    print('Effect size: Cohen\'s d.s = {:.3g}; benchmarks |d.s|: 0.2 = small, 0.5 = medium, 0.8 = large'.format(d_s))
    print(80*'-' + '\n')
    return(t, p_value, t_crit_L, t_crit_R)


# ## 8. 2-sample (Welch's) $t$-test (1- and 2-sided)

# In[25]:


def DS_2sample_ttest_means(y1, y2, equal_var=False, alternative='two-sided', alpha=0.05):
    """
    *
    Function DS_2sample_ttest_means(y1, y2, equal_var=False, alternative='two-sided', alpha=0.05)
    
       This function performs a 2-sample (Welch's) t-test (Null Hypothesis Significance Test) 
       in the spirit of R, testing 2 averages with *unknown* standard deviation.
       The function also evaluates the effect size (Cohen's d).
       
    Requires:          -
       
    Usage:             DS_2sample_ttest_means(y1, y2, 
                            alternative=['two-sided']/'less'/'greater',
                            equal_var=[False]/True, alpha = 0.05)
     
                         alternative = 'two-sided' [default]  H1: mu_1 != mu_2
                                       'less'                 H1: mu_1 < mu_2
                                       'greater'              H1: mu_1 > mu_2
                         equal_var = False                    perform Welch t-test
                                     True                     perform 2-sample t-test
                         alpha:   significance level of test [default: 0.05]
     
    Return:            t, p-value, t.crit.L, t.crit.R  [ + print interpretable output to stdout ]
                       where t.crit.L and t.crit.R are the lower and upper critical values, 
                       t is the test statistic and p-value is the p-value of the test.     
     
    Author:            M.E.F. Apol
    Date:              2022-01-28, rev. 2022_08_26
    Validation:
    """
    
    from scipy.stats import ttest_ind
    from scipy.stats import t as t_distr
    import numpy as np
    
    t, p_samp = ttest_ind(y1, y2, equal_var = equal_var)
    y_av_1 = np.mean(y1)
    y_av_2 = np.mean(y2)
    n_1 = len(y1)
    n_2 = len(y2)
    s2_1 = np.var(y1, ddof=1)
    s2_2 = np.var(y2, ddof=1)
    print(80*'-')
    if equal_var == True:
        print('2-sample t-test for 2 means:')
        print('     assuming Normal(mu.1, sigma2) data for dataset 1')
        print('     assuming Normal(mu.2, sigma2) data for dataset 2')
        df = n_1 + n_2 - 2
    else:
        print('Welch t-test for 2 means:')
        df = (s2_1/n_1 + s2_2/n_2)**2 / ( 1/(n_1-1)*(s2_1/n_1)**2 + 1/(n_2-1)*(s2_2/n_2)**2 )
        print('     assuming Normal(mu.1, sigma2.1) data for dataset 1')
        print('     assuming Normal(mu.2, sigma2.2) data for dataset 2')
    print('y.av.1 = {:.3g}, y.av.2 = {:.3g}, s2.1 = {:.3g}, s2.2 = {:.3g}, n.1 = {:d}, n.2 = {:d}, alpha = {:.3g}'.format(y_av_1, y_av_2, s2_1, s2_2, n_1, n_2, alpha))
    print('H0: mu.1  = mu.2')
    if alternative == 'two-sided':
        print('H1: mu.1 != mu.2')
        p_value = p_samp
        t_crit_L = t_distr.ppf(alpha/2, df)
        t_crit_R = t_distr.ppf(1-alpha/2, df)      
    elif alternative == 'less':
        print('H1: mu.1  < mu.2')
        if t <= 0:
            p_value = p_samp/2
        else:
            p_value = 1 - p_samp/2
        t_crit_L = t_distr.ppf(alpha, df)
        t_crit_R = float('inf')
    elif alternative == 'greater':
        print('H1: mu.1  > mu.2')
        if t >= 0:
            p_value = p_samp/2
        else:
            p_value = 1 - p_samp/2
        t_crit_L = float('-inf')
        t_crit_R = t_distr.ppf(1-alpha, df)
    else:
        print('Wrong alternative hypothesis chosen!')
        print(80*'-' + '\n')
        t, p_value, t_crit_L, t_crit_R = np.nan, np.nan, np.nan, np.nan
        return(t, p_value, t_crit_L, t_crit_R)
    
    # Effect size (Cohen's d.av):
    d_av = t * np.sqrt(1/n_1 + 1/n_2)
    print('t = {:.4g}, p-value = {:.4g}, t.crit.L = {:.4g}, t.crit.R = {:.4g}, df = {:.4g}'.format(t, p_value, t_crit_L, t_crit_R, df))
    print(80*'.')
    print('Effect size: Cohen\'s d.s = {:.3g}; benchmarks |d.s|: 0.2 = small, 0.5 = medium, 0.8 = large'.format(d_av))
    print(80*'-' + '\n')
    return(t, p_value, t_crit_L, t_crit_R)


# ## 9. Paired $t$-test (1- and 2-sided)

# In[28]:


def DS_paired_ttest_means(y1, y2, alternative='two-sided', alpha=0.05):
    """
    *
    Function DS_paired_ttest_means(y1, y2, alternative='two-sided', alpha=0.05)
     
       This function performs a paired t-test (Null Hypothesis Significance Test) 
       in the spirit of R, testing 2 averages of paired data with *unknown* standard deviation.
       The function also evaluates the effect size (Cohen's d).
       
    Requires:          -
       
    Usage:             DS_paired_ttest_means(y1, y2, 
                            alternative=['two-sided']/'less'/'greater', 
                            alpha = 0.05)
     
                         alternative = 'two-sided' [default]  H1: mu_1 != mu_2
                                       'less'                 H1: mu_1 < mu_2
                                       'greater'              H1: mu_1 > mu_2
                         alpha:   significance level of test [default: 0.05]
     
    Return:            t, p-value, t.crit.L, t.crit.R  [ + print interpretable output to stdout ]
                       where t.crit.L and t.crit.R are the lower and upper critical values, 
                       t is the test statistic and p-value is the p-value of the test.     
    
    Author:            M.E.F. Apol
    Date:              2020-11-11, rev. 2022_08_26
    Validation:
    """
    
    from scipy.stats import ttest_rel
    from scipy.stats import t as t_distr
    import numpy as np
    
    # Error checking:
    if len(y1) != len(y2):
        print('Error: Datasets of unequal length...')
        return
    
    t, p_samp = ttest_rel(y1, y2)
    y_av_1 = np.mean(y1)
    y_av_2 = np.mean(y2)
    d_av = y_av_1 - y_av_2
    n_d = len(y1)
    df = n_d - 1
    s2_1 = np.var(y1, ddof=1)
    s2_2 = np.var(y2, ddof=1)
    print(80*'-')
    print('Paired t-test for 2 means:')
    print('     assuming Normal(mu.d, sigma2.d) data for difference between datasets 1 and 2')
    print('y.av.1 = {:.3g}, y.av.2 = {:.3g}, s2.1 = {:.3g}, s2.2 = {:.3g}, n.1 = {:d}, n.2 = {:d}, alpha = {:.3g}'.format(y_av_1, y_av_2, s2_1, s2_2, n_d, n_d, alpha))
    print('H0: mu.1  = mu.2')
    if alternative == 'two-sided':
        print('H1: mu.1 != mu.2')
        p_value = p_samp
        t_crit_L = t_distr.ppf(alpha/2, df)
        t_crit_R = t_distr.ppf(1-alpha/2, df)
    elif alternative == 'less':
        print('H1: mu.1  < mu.2')
        if t <= 0:
            p_value = p_samp/2
        else:
            p_value = 1 - p_samp/2
        t_crit_L = t_distr.ppf(alpha, df)
        t_crit_R = float('inf')
    elif alternative == 'greater':
        print('H1: mu.1  > mu.2')
        if t >= 0:
            p_value = p_samp/2
        else:
            p_value = 1 - p_samp/2
        t_crit_L = float('-inf')
        t_crit_R = t_distr.ppf(1-alpha, df)
    else:
        print('Wrong alternative hypothesis chosen!')
        print(80*'-' + '\n')
        t, p_value, t_crit_L, t_crit_R = np.nan, np.nan, np.nan, np.nan
        return(t, p_value, t_crit_L, t_crit_R)
    # Effect size:
    d_av = (y_av_1 - y_av_2) / np.sqrt((s2_1 + s2_2)/2)
    print('t = {:.4g}, p-value = {:.4g}, t.crit.L = {:.4g}, t.crit.R = {:.4g}, df = {:.4g}'.format(t, p_value, t_crit_L, t_crit_R, df))
    print(80*'.')
    print('Effect size: Cohen\'s d.av = {:.3g}; benchmarks |d.av|: 0.2 = small, 0.5 = medium, 0.8 = large'.format(d_av))
    print(80*'-' + '\n')
    return(t, p_value, t_crit_L, t_crit_R)


# ## <span style="color:blue">*Tests for variances (assuming normal distribution):*</span>

# ## 10. $\chi^2$-test for one variance (1- and 2-sided)

# In[31]:


def DS_1sample_chi2test_vars(y, sigma, alternative = 'two-sided', alpha = 0.05):
    """
    *
    Function DS_1sample_chi2test_vars(y, sigma, alternative = 'two-sided', alpha = 0.05)
         
       This function performs a 1-sample chi2-test (Null Hypothesis Significance Test)
       in the spirit of R, testing 1 variance sigma^2 (or standard deviation sigma)
       using a normal approximation. 
    
    Requires:          -
    
    Usage:             DS_1sample_chi2test_vars(y, sigma, 
                                         alternative=['two-sided']/'less'/'greater', alpha=0.05)
                         sigma    reference standard deviation sigma*
                         alternative = 'two-sided' [default]  H1: sigma^2 != sigma*^2
                                       'less'                 H1: sigma^2 < sigma*^2
                                       'greater'              H1: sigma^2 > sigma*^2
                         alpha:   significance level of test [default: 0.05]
     
    Return:            chi2, p-value, chi2.crit.L, chi2.crit.R  [ + print interpretable output to stdout ]
                       where chi2.crit.L and chi2.crit.R are the lower and upper critical values, 
                       chi2 is the test statistic and p-value is the p-value of the test.    
     
    Author:            M.E.F. Apol
    Date:              2022-01-31, rev. 2022_08_26
    Validation:        2022-01-31 against Minitab 19  
    
    """
    
    from scipy.stats import chi2
    import numpy as np
        
    n = len(y)
    y_av = np.mean(y)
    s2 = np.var(y, ddof=1)
    
    print(80*'-')
    print('Chi2-test for 1 variance:')
    print('     assuming Normal(mu, sigma2) data for dataset')
    print('s2 = {:.3g}, sigma*2 = {:.3g}, n = {:d}, alpha = {:.3g}'.format(s2, sigma**2, n, alpha))
    print('H0: sigma2  = sigma*2')
    if alternative == 'two-sided':
        print('H1: sigma2 != sigma*2')
    elif alternative == 'greater':
        print('H1: sigma2  > sigma*2')
    elif alternative == 'less':
        print('H1: sigma2  < sigma*2')
    else:
        print('Wrong alternative hypothesis chosen!')
        print(80*'-' + '\n')
        chi2_samp, p_value, chi2_crit_L, chi2_crit_R = np.nan, np.nan, np.nan, np.nan
        return(chi2_samp, p_value, chi2_crit_L, chi2_crit_R)
        
    df = n - 1
    chi2_samp = df*s2 / sigma**2
    if alternative == 'two-sided':
        p_value = 2*np.min([chi2.cdf(chi2_samp, df), chi2.sf(chi2_samp, df)])
        chi2_crit_L = chi2.ppf(alpha/2, df)
        chi2_crit_R = chi2.ppf(1-alpha/2, df)
    elif alternative == 'greater':
        p_value = 1-chi2.cdf(chi2_samp, df)
        # better precision: use survival function
        p_value = chi2.sf(chi2_samp, df)
        chi2_crit_L = 0
        chi2_crit_R = chi2.ppf(1-alpha, df)
    elif alternative == 'less':
        p_value = chi2.cdf(chi2_samp, df)
        chi2_crit_L = chi2.ppf(alpha, df)
        chi2_crit_R = float('inf')
    
    print('chi2 = {:.4g}, p-value = {:.4g}, chi2.crit.L = {:.4g}, chi2.crit.R = {:.4g}'.format(chi2_samp, p_value, chi2_crit_L, chi2_crit_R))
    print(80*'-' + '\n')
    return(chi2_samp, p_value, chi2_crit_L, chi2_crit_R)     


# ## 11. $F$-test for 2 variances (1- and 2-sided)

# In[36]:


def DS_2sample_Ftest_vars(y1, y2, alternative = 'two-sided', alpha = 0.05):
    """
    *
    Function DS_2sample_Ftest_vars(y1, y2, alternative='two-sided', alpha=0.05)
     
       This function performs a 2-sample F-test (Null Hypothesis Significance Test)
       in the spirit of R, testing 2 variances using a normal approximation. 
       Note that there are *two different conventions* to define the test statistic F:
       1) F = (expected) largest variance / (expected) smallest variance, 
          see Miller & Miller (2005) - Statistics and Chemometrics for Analytical Chemistry. 
          5th ed. Pearson. 
          In that case, F >= 1, and the the left critical F-value, F.crit.L is undefined.
       2) F = s.1^2 / s.2^2, so F may be smaller or larger then 1.
       The output of the function uses convention 1; the output to stdout gives both conventions!
       The p-value of the test is - following Fisher - 2 times the upper tail probability 
       using convention 1.
    
    Requires:          -
    
    Usage:             DS_2sample_Ftest_vars(y1, y2,
                                         alternative=['two-sided']/'less'/'greater', alpha=0.05)
                         alternative = 'two-sided' [default]  H1: sigma_1^2 != sigma_2^2
                                       'less'                 H1: sigma_1^2 < sigma_2^2
                                       'greater'              H1: sigma_1^2 > sigma_2^2
                         alpha:   significance level of test [default: 0.05]
     
    Return:            F, p-value, F.crit.L, F.crit.R  [ + print interpretable output to stdout ]
                       where F.crit.L and F.crit.R are the lower and upper critical values, 
                       F is the test statistic and p-value is the p-value of the test.    
     
    Author:            M.E.F. Apol
    Date:              2022-01-30, rev. 2022_08_26
    Validation:        2022-01-31 against Minitab 19
    """
        
    from scipy.stats import f
    import numpy as np
        
    n_1 = len(y1); n_2 = len(y2)
    y_av_1 = np.mean(y1); y_av_2 = np.mean(y2)
    s2_1 = np.var(y1, ddof=1); s2_2 = np.var(y2, ddof=1)
    
    print(80*'-')
    
    # Perform F-test with convention that F = s2.1 / s2.2
    
    print('2-sample F-test for 2 variances:')
    print('     assuming Normal(mu.1, sigma2.1) data for dataset 1')
    print('     assuming Normal(mu.2, sigma2.2) data for dataset 2')
    print('s2.1 = {:.3g}, s2.2 = {:.3g}, n.1 = {:d}, n.2 = {:d}, alpha = {:.3g}'.format(s2_1, s2_2, n_1, n_2, alpha))
    print('H0: sigma2.1  = sigma2.2')
    if alternative == 'two-sided':
        print('H1: sigma2.1 != sigma2.2')
    elif alternative == 'greater':
        print('H1: sigma2.1  > sigma2.2')
    elif alternative == 'less':
        print('H1: sigma2.1  < sigma2.2')
    else:
        print('Wrong alternative hypothesis chosen!')
        print(80*'-' + '\n')
        F_samp, p_value, F_crit_L, F_crit_R = np.nan, np.nan, np.nan, np.nan
        return(F_samp, p_value, F_crit_L, F_crit_R)
    print(80*'.')
    
    # Perform F-test with convention that F = (expected) largest s2 / (expected) smallest s2 (Miller & Miller)
    
    print('     using convention that F = (expected) largest s2 / (expected) smallest s2:')   
    if alternative == 'two-sided':
        if (s2_1 > s2_2):
            s2_max = s2_1; nu_1 = n_1 - 1
            s2_min = s2_2; nu_2 = n_2 - 1
        else:
            s2_max = s2_2; nu_1 = n_2 - 1
            s2_min = s2_1; nu_2 = n_1 - 1
        F_samp = s2_max / s2_min
        p_value = 2*(1-f.cdf(F_samp, nu_1, nu_2))
        # better precision: use survival function
        p_value = 2*f.sf(F_samp, nu_1, nu_2)
        F_crit_L = float('NaN')
        F_crit_R = f.ppf(1-alpha/2, nu_1, nu_2)
    if alternative == 'greater':       
        F_samp = s2_1 / s2_2
        nu_1 = n_1 - 1; nu_2 = n_2 - 1
        p_value = 1-f.cdf(F_samp, nu_1, nu_2)
        # better precision: use survival function
        p_value = f.sf(F_samp, nu_1, nu_2)
        F_crit_L = float('NaN')
        F_crit_R = f.ppf(1-alpha, nu_1, nu_2)
    if alternative == 'less':
        F_samp = s2_2 / s2_1
        nu_1 = n_2 - 1; nu_2 = n_1 - 1
        p_value = 1-f.cdf(F_samp, nu_1, nu_2)
        # better precision: use survival function
        p_value = f.sf(F_samp, nu_1, nu_2)
        F_crit_L = float('NaN')
        F_crit_R = f.ppf(1-alpha, nu_1, nu_2)
    print('F = {:.4g}, p-value = {:.4g}, F.crit.L = {:.4g}, F.crit.R = {:.4g}, df.1 = {:.4g}, df.2 = {:.4g}'.format(F_samp, p_value, F_crit_L, F_crit_R, nu_1, nu_2))
    print(80*'.')
    
    # Perform F-test with convention that F = s2.1 / s2.2
    
    print('     using convention that F = s2.1/s2.2:')
    nu_12 = n_1 - 1
    nu_22 = n_2 - 1
    F_samp2 = s2_1 / s2_2
    if alternative == 'two-sided':
        p_value2 = 2*np.min([f.cdf(F_samp2, nu_12, nu_22), f.sf(F_samp2, nu_12, nu_22)])
        F_crit_L2 = f.ppf(alpha/2, nu_12, nu_22)
        F_crit_R2 = f.ppf(1-alpha/2, nu_12, nu_22)
    if alternative == 'greater':
        p_value2 = 1-f.cdf(F_samp2, nu_12, nu_22)
        # better precision: use survival function
        p_value2 = f.sf(F_samp2, nu_12, nu_22)
        F_crit_L2 = 0
        F_crit_R2 = f.ppf(1-alpha, nu_12, nu_22)
    if alternative == 'less':
        p_value2 = f.cdf(F_samp2, nu_12, nu_22)
        F_crit_L2 = f.ppf(alpha, nu_12, nu_22)
        F_crit_R2 = float('inf')
    print('F = {:.4g}, p-value = {:.4g}, F.crit.L = {:.4g}, F.crit.R = {:.4g}, df.1 = {:.4g}, df.2 = {:.4g}'.format(F_samp2, p_value2, F_crit_L2, F_crit_R2, nu_12, nu_22))
    print(80*'-' + '\n')
    return(F_samp, p_value, F_crit_L, F_crit_R)     


# ## 12. Levene-Brown-Forsythe test for 2 variances (1- and 2-sided)

# In[40]:


def DS_2sample_Levenetest_vars(y1, y2, center = 'median', alternative = 'two-sided', alpha = 0.05):
    """
    *
    Function DS_2sample_Levenetest_vars(y1, y2, center='median', alternative='two-sided', alpha=0.05)
     
       This function performs a 2-sample Levene-test (Null Hypothesis Significance Test)
       in the spirit of R, testing 2 variances without using a normal approximation. 
    
    Requires:          -
    
    Usage:             DS_2sample_Levenetest_vars(y1, y2, center = ['median']/'mean', 
                                         alternative=['two-sided']/'less'/'greater', alpha=0.05)
    
                         center = 'median' [default]          use z.ik = |y.ik - median(y.i)| (Brown-Forsythe)
                                  'mean'                      use z.ik = |y.ik - mean(y.i)|   (Levene)
                         alternative = 'two-sided' [default]  H1: sigma_1^2 != sigma_2^2
                                       'less'                 H1: sigma_1^2 < sigma_2^2
                                       'greater'              H1: sigma_1^2 > sigma_2^2
                         alpha:   significance level of test [default: 0.05]
     
    Return:            t, p-value, t.crit.L, t.crit.R  [ + print interpretable output to stdout ]
                       where t.crit.L and t.crit.R are the lower and upper critical values, 
                       t is the test statistic and p-value is the p-value of the test.    
     
    Author:            M.E.F. Apol
    Date:              2022-02-01, rev. 2022_08_26
    Validation:        2022-02-01 against scipy.statst.levene and https://www.socscistatistics.com/
    """
     
    from scipy.stats import t as t_dist
    import numpy as np
        
    n_1 = len(y1);               n_2 = len(y2)
    y_av_1  = np.mean(y1);    y_av_2 = np.mean(y2)
    y_med_1 = np.median(y1); y_med_2 = np.median(y2)
    s2_1 = np.var(y1, ddof=1);  s2_2 = np.var(y2, ddof=1) 
    
    print(80*'-')
    if center == 'median':
        print('Brown-Forsythe test for 2 variances:')
        print('     using the median of both datasets as center')
        y_c_1 = y_med_1; y_c_2 = y_med_2
    elif center == 'mean':
        print('Levene test for 2 variances:')
        print('     using the mean of both datasets as center')
        
        y_c_1 = y_av_1; y_c_2 = y_av_2
    else:
        print('Wrong center option...')
        print(80*'-' + '\n')
        t_samp, p_value, t_crit_L, t_crit_R = np.nan, np.nan, np.nan, np.nan
        return(t_samp, p_value, t_crit_L, t_crit_R)
    
    print('s2.1 = {:.3g}, s2.2 = {:.3g}, n.1 = {:d}, n.2 = {:d}, alpha = {:.3g}'.format(s2_1, s2_2, n_1, n_2, alpha))
    print('H0: sigma.1  = sigma.2')
    if alternative == 'two-sided':
        print('H1: sigma.1 != sigma.2')
    elif alternative == 'greater':
        print('H1: sigma.1  > sigma.2')
    elif alternative == 'less':
        print('H1: sigma.1  < sigma.2')
    else:
        print('Wrong alternative hypothesis chosen!')
        print(80*'-' + '\n')
        t_samp, p_value, t_crit_L, t_crit_R = np.nan, np.nan, np.nan, np.nan
        return(t_samp, p_value, t_crit_L, t_crit_R)
    
    z_1 = np.abs(y1 - y_c_1)
    z_2 = np.abs(y2 - y_c_2)
    z_av_1 = np.mean(z_1);        z_av_2 = np.mean(z_2)
    s2_z_1 = np.var(z_1, ddof=1); s2_z_2 = np.var(z_2, ddof=1)
    df = n_1 + n_2 - 2
    s2_pooled = ((n_1 - 1)*s2_z_1 + (n_2 - 1)*s2_z_2)/df
    t_samp = (z_av_1 - z_av_2) / np.sqrt(s2_pooled * (1/n_1 + 1/n_2))
    F = t_samp**2
    
    if alternative == 'two-sided':
        p_value = 2*(t_dist.cdf(-np.abs(t_samp), df))
        t_crit_L = t_dist.ppf(alpha/2, df)
        t_crit_R = t_dist.ppf(1-alpha/2, df)
    elif alternative == 'greater':
        p_value = 1-t_dist.cdf(t_samp, df)
        # better precision: use survival function
        p_value = t_dist.sf(t_samp, df)
        t_crit_L = float('-inf')
        t_crit_R = t_dist.ppf(1-alpha, df)
    elif alternative == 'less':
        p_value = t_dist.cdf(t_samp, df)
        t_crit_L = t_dist.ppf(alpha, df)
        t_crit_R = float('inf')
    print('t = {:.4g}, p-value = {:.4g}, t.crit.L = {:.4g}, t.crit.R = {:.4g}, F = {:.4g}'.format(t_samp, p_value, t_crit_L, t_crit_R, F))
    print(80*'-' + '\n')
    return(t_samp, p_value, t_crit_L, t_crit_R)     


# ## 13. Pitman-Morgan test for variances of paired data

# In[44]:


def DS_paired_PitmanMorgan_test_vars(y1, y2, alternative='two-sided', alpha=0.05):
    
    from scipy.stats import pearsonr
    from scipy.stats import t as t_dist
    
    n_1 = len(y1); n_2 = len(y2)
    if(n_1 != n_2):
        raise ValueError('Datasets of unequal length!')
    
    
    n = n_1
    s2_1 = np.var(y1, ddof=1)
    s2_2 = np.var(y2, ddof=1)
    r = pearsonr(y1, y2)[0]
    # First, calculate sums and differences between y.1 and y.2:
    u = y1 + y2
    v = y1 - y2
    
    r_uv = pearsonr(u, v)[0]
    t_uv = r_uv * np.sqrt((n-2) / (1 - r_uv**2))
    df_uv = n - 2
    F_uv = t_uv**2
    
    print(r_uv, t_uv, df_uv)
        
    print(80*'-')
    print('Paired Pitman-Morgan-test for 2 variances:')
    print('     assuming y1, y2 ~ bivariate normal')
    print('s2.1 = {:.4g}, s2.2 = {:.4g}, r = {:.4g}, n.1 = {:d}, n.2 = {:d}, alpha = {:.3g}'.format(s2_1, s2_2, r, n_1, n_2, alpha))
    print('H0: sigma2.1  = sigma2.2')
     
    
    if alternative == 'two-sided':
        print('H1: sigma2.1 != sigma2.2')
        p_value = 2*(t_dist.cdf(-np.abs(t_uv), df_uv))
        t_crit_L = t_dist.ppf(alpha/2, df_uv)
        t_crit_R = t_dist.ppf(1-alpha/2, df_uv)
    elif alternative == 'greater':
        print('H1: sigma2.1  > sigma2.2')
        p_value = 1-t_dist.cdf(t_uv, df_uv)
        # better precision: use survival function
        p_value = t_dist.sf(t_uv, df_uv)
        t_crit_L = float('-inf')
        t_crit_R = t_dist.ppf(1-alpha, df_uv)
    elif alternative == 'less':
        print('H1: sigma2.1  < sigma2.2')
        p_value = t_dist.cdf(t_uv, df_uv)
        t_crit_L = t_dist.ppf(alpha, df_uv)
        t_crit_R = float('inf')
    print('t = {:.4g}, p-value = {:.4g}, t.crit.L = {:.4g}, t.crit.R = {:.4g}'.format(t_uv, p_value, t_crit_L, t_crit_R))
    print(80*'-' + '\n')
        
    return(t_uv, p_value, t_crit_L, t_crit_R);


# ## <span style="color:blue">*Tests for medians (not assuming a normal distribution):*</span>

# ## 14. 1-sample Wilcoxon-test for median (1-and 2-sided)

# In[48]:


def DS_1sample_Wilcoxon_test_medians(y, popmedian = 0, alternative='two-sided', alpha=0.05):
    """
    *
    Function DS_1sample_Wilcoxon_test_medians(y, popmedian, alternative='two-sided', alpha=0.05)
    
       This function tests the median of y (eta) against a reference value eta* (Null Hypothesis Significance Test).
       The distribution is assumed to be symmetrical, but not necessarily normal.
    
    Requires:            scipy.stats.wilcoxon
    
    Usage:               DS_1sample_Wilcoxon_test_medians(y, popmedian, 
                              alternative=['two-sided']/'less'/'greater', alpha=0.05)
    
    Arguments:
      y                  data array
      popmedian          reference value of the median, median*
      alternative        'two-sided' [default]   H1: eta != eta*
                         'less'                  H1: eta <  eta*
                         'greater'               H1: eta >  eta*
      alpha              significance level of test [default: 0.05]     
 
 
    Returns:             T, p_value [ + print interpretable output to stdout ]
    where
      T                  Wilcoxon statistic
      p_value            p-value of Wilcoxon signed-rank-test
      
    Author:            M.E.F. Apol
    Date:              2023-12-14, update 2023-12-19
    """
    
    from scipy.stats import wilcoxon
    from scipy.stats import iqr
    
    # Additional statistics:
    y_med = np.median(y)
    iqr = iqr(y)
    n = len(y)
    
    print(80*'-')
    print('Wilcoxon 1-sample signed-rank-test for 1 median:')
    print('     assuming symmetrical distribution for dataset')
    print('y.med = {:.4g}, eta* = {:.4g}, iqr = {:.4g}, n = {:d}, alpha = {:.3g}'.format(y_med, popmedian, iqr, n, alpha))
    print('H0: eta  = eta*')
    
    if alternative == 'two-sided':
        print('H1: eta != eta*')
    elif alternative == 'greater':
        print('H1: eta  > eta*')
    elif alternative == 'less':
        print('H1: eta  < eta*')
    else:
        print('Wrong alternative hypothesis chosen!')
        print(80*'-' + '\n')
        T, p_value = np.nan, np.nan
        return(T, p_value)
    
    res = wilcoxon(y-popmedian, zero_method='wilcox', correction=True, 
                 alternative=alternative)
    T = res.statistic
    p_value = res.pvalue
    
    # -- TO DO -- Find formula for T.crit
    T_crit = np.nan
    
    # To compute an effect size for the signed-rank test, one can use the rank-biserial correlation?
    # -- TO DO --
    
    print('T = {:.3g}, p-value = {:.3g}, T.crit = {:.3g}'.format(T, p_value, T_crit))
    #print(80*'.')
    #print effect size
    print(80*'-')
    
    return(T, p_value);


# ## 15. 2-sample Mann-Whitney-test for medians (1-and 2-sided)

# In[52]:


def DS_2sample_MannWhitney_test_medians(y1, y2, alternative='two-sided', alpha=0.05):
    """
    *
    Function DS_2sample_MannWhitney_test_medians(y1, y2, alternative='two-sided', alpha=0.05)
    
       This function tests two medians eta.1 and eta.2 of data y1 and y2 (Null Hypothesis Significance Test).
       The distributions are assumed to be identical, but not necessarily normal.
    
    Requires:            scipy.stats.mannwhitneyu
    
    Usage:               DS_2sample_MannWhitney_test_medians(y1, y2, 
                              alternative=['two-sided']/'less'/'greater', alpha=0.05)
    
    Arguments:
      y1, y2             data arrays
      alternative        'two-sided' [default]   H1: eta.1 != eta.2
                         'less'                  H1: eta.1 <  eta.2
                         'greater'               H1: eta.1 >  eta.2
      alpha              significance level of test [default: 0.05]     
 
 
    Returns:             U.1, U.2, U, p_value [ + print interpretable output to stdout ]
    where
      U.1, U.2, U        Mann-Whitney statistics
      p_value            p-value of Mann-Whitney U-test
      
    Validation:          against SPSS v. 28
      
    Author:            M.E.F. Apol
    Date:              2023-12-20, revision 2024_01_02
    """

    from scipy.stats import mannwhitneyu
    
    # Additional statistics:
    y_med_1 = np.median(y1)
    y_med_2 = np.median(y2)
    n_1 = len(y1)
    n_2 = len(y2)
    
    print(80*'-')
    print('2-sample Mann-Whitney U-test for 2 medians:')
    print('     assuming identical distributions for both datasets, that may differ in location')
    print('y.med.1 = {:.4g}, y.med.2 = {:.4g}, n.1 = {:d}, n.2 = {:d}, alpha = {:.3g}'.format(y_med_1, y_med_2, n_1, n_2, alpha))
    print('H0: eta.1  = eta.2')
    
    if alternative == 'two-sided':
        print('H1: eta != eta*')
    elif alternative == 'greater':
        print('H1: eta  > eta*')
    elif alternative == 'less':
        print('H1: eta  < eta*')
    else:
        print('Wrong alternative hypothesis chosen!')
        print(80*'-' + '\n')
        U_1, U_2, U, p_value = np.nan, np.nan, np.nan, np.nan
        return(U_1, U_2, U, p_value)
    
    #res = mannwhitneyu(y1, y2, alternative=alternative, use_continuity=True)
    res = mannwhitneyu(y1, y2, alternative=alternative, use_continuity=False)
    U_1 = res.statistic
    U_2 = n_1*n_2 - U_1
    U = np.min([U_1, U_2])
    p_value = res.pvalue
    
    # -- TO DO -- Find formula for U.crit
    U_crit = np.nan
    
    # To compute an effect size for the signed-rank test, one can use the rank-biserial correlation?
    
    # Correlation coefficient:
    # From: datatab.net (dd 2023_12_19), https://maths.shu.ac.uk/mathshelp/ (dd 2023_12_19)
    # Normal approximation:
    # Expectation value:
    mu_U = n_1*n_2/2
    # Standard deviation:
    sigma_U = np.sqrt(n_1*n_2*(n_1+n_2+1)/12)
    # Standard normal statistic:
    z = (U - mu_U) / sigma_U
    # Effect size:
    r = z / np.sqrt(n_1+n_2)
    
    # Point biserial correlation r.pb
    # From: https://www.andrews.edu/~calkins/math/edrm611/edrm13.htm
    # Additional statistics:
    y_av_1 = np.mean(y1)
    y_av_2 = np.mean(y2)
    s2_1 = np.var(y1, ddof=1)
    s2_2 = np.var(y2, ddof=1)
    s2_p = ((n_1-1)*s2_1 + (n_2-1)*s2_2)/(n_1+n_2-2)
    s_p = np.sqrt(s2_p)
    r_pb = (y_av_1 - y_av_2)/s_p * np.sqrt(n_1*n_2)/(n_1 + n_2)
    
    print('U.1 = {:.3g}, U.2 = {:.3g}, U = {:.3g}, p-value = {:.3g}, z = {:.3g}, U.crit = {:.3g}'.format(U_1, U_2, U, p_value, z, U_crit))
    print(80*'.')
    print('Effect size: r    = {:.3g}; benchmarks |r|: 0.1 = small, 0.3 = medium, 0.5 = large'.format(r))
    # print('Effect size: r.pb = {:.3g}; benchmarks r: 0.1 = small, 0.3 = medium, 0.5 = large (?)'.format(r_pb))
    print(80*'-')
    
    return(U_1, U_2, U, p_value);


# ## 16. Paired Wilcoxon test for 2 medians

# In[59]:


def DS_paired_Wilcoxon_test_medians(y1, y2, alternative='two-sided', alpha=0.05):
    """
    *
    Function DS_paired_Wilcoxon_test_medians(y1, y2, alternative='two-sided', alpha=0.05)
    
       This function tests two medians eta.1 and eta.2 of paired data y1 and y2 (Null Hypothesis Significance Test).
       The distributions are assumed to be symmetrical, but not necessarily normal.
    
    Requires:            scipy.stats.wilcoxon, scipy.stats.norm
    
    Usage:               DS_paired_Wilcoxon_test_medians(y1, y2, 
                              alternative=['two-sided']/'less'/'greater', alpha=0.05)
    
    Arguments:
      y1, y2             data arrays
      alternative        'two-sided' [default]   H1: eta.1 != eta.2
                         'less'                  H1: eta.1 <  eta.2
                         'greater'               H1: eta.1 >  eta.2
      alpha              significance level of test [default: 0.05]     
 
 
    Returns:             W, p_value [ + print interpretable output to stdout ]
    where
      T                  Wilcoxon signed-rank statistic
      p_value            p-value of Wilcoxon signed-rank-test
      
    Validation:          against Minitab 21 (2023-12-26)
      
    Author:              M.E.F. Apol
    Date:                2023-12-26, revision 2024_01_02
    """

    from scipy.stats import wilcoxon
    from scipy.stats import norm
    
    # Additional statistics:
    y_med_1 = np.median(y1)
    y_med_2 = np.median(y2)
    n_1 = len(y1)
    n_2 = len(y2)
    
    if(n_1 != n_2):
        print('Datasets of unequal length!')
        T, p_value = np.nan, np.nan
    else:
        n = n_1
        # However, it must be the number of non-identical pairs
        # (i.e., pairs with difference != 0)
        d = y1 - y2   # differences
        rd = d[np.nonzero(d)]
        n = len(rd)
        
        print(80*'-')
        print('Paired Wilcoxon signed-rank-test for 2 medians:')
        print('     assuming symmetrical distributions for both datasets, that may differ in location')
        print('y.med.1 = {:.4g}, y.med.2 = {:.4g}, n.1 = {:d}, n.2 = {:d}, n (test) = {:d}, alpha = {:.3g}'.format(y_med_1, y_med_2, n_1, n_2, n, alpha))
        print('H0: eta.1  = eta.2')
    
        if alternative == 'two-sided':
            print('H1: eta != eta*')
        elif alternative == 'greater':
            print('H1: eta  > eta*')
        elif alternative == 'less':
            print('H1: eta  < eta*')
        else:
            print('Wrong alternative hypothesis chosen!')
            print(80*'-' + '\n')
            W, p_value = np.nan, np.nan
            return(W, p_value)
    
        res = wilcoxon(y1, y2, alternative=alternative, correction=True, zero_method='wilcox')
        W = res.statistic
        p_value = res.pvalue
        
        # -- TO DO -- Find formula for W.crit
        W_crit = np.nan
        
        # To compute an effect size for the signed-rank test, one can use the rank-biserial correlation?
        
        # Correlation coefficient:
        
        # From: Minitab 21 help (dd 2023-12-26)
        # Normal approximation:
        # Expectation value:
        mu_W = n*(n+1)/4
        # Standard deviation:
        sigma_W = np.sqrt(n*(n+1)*(2*n+1)/24)
        # Standard normal statistic:
        z = (W - mu_W) / sigma_W
        # Effect size:
        r = z / np.sqrt(n)
        
        # Also perform continuity correaction (Minitab 21)
        if alternative == 'two-sided':
            k = np.min([W, n*(n+1)/2 - W])
            z_t = (k + 0.5 - mu_W)/sigma_W
            #p_value_approx = 2*norm.cdf(-np.abs(z), 0, 1)
            p_value_approx = 2*norm.cdf(z_t, 0, 1)
        elif alternative == 'greater':
            z_g = (W - 0.5 - mu_W)/sigma_W
            p_value_approx = 1 - norm.cdf(z_g, 0, 1)
            # better accuracy:
            p_value_approx = norm.sf(z_g, 0, 1)
        elif alternative == 'less':
            z_l = (W + 0.5 - mu_W)/sigma_W
            p_value_approx = norm.cdf(z_l, 0, 1)
        
        # Point biserial correlation r.pb
        # From: https://www.andrews.edu/~calkins/math/edrm611/edrm13.htm
        # Additional statistics:
        #y_av_1 = np.mean(y1)
        #y_av_2 = np.mean(y2)
        #s2_1 = np.var(y1, ddof=1)
        #s2_2 = np.var(y2, ddof=1)
        #s2_p = ((n_1-1)*s2_1 + (n_2-1)*s2_2)/(n_1+n_2-2)
        #s_p = np.sqrt(s2_p)
        #r_pb = (y_av_1 - y_av_2)/s_p * np.sqrt(n_1*n_2)/(n_1 + n_2)
        
        print('W = {:.3g}, p-value = {:.3g}, z = {:.3g}, W.crit = {:.3g}, p-value (approx) = {:.3g}'.format(W, p_value, z, W_crit, p_value_approx))
        print(80*'.')
        print('Effect size(?): r    = {:.3g}; benchmarks |r|: 0.1 = small, 0.3 = medium, 0.5 = large'.format(r))
        # print('Effect size: r.pb = {:.3g}; benchmarks r: 0.1 = small, 0.3 = medium, 0.5 = large (?)'.format(r_pb))
        print(80*'-')
    
        return(W, p_value);
    pass;


# ## <span style="color:blue">*Tests for proportions (assuming a Bernoulli or Binomial distribution):*</span>

# ## 17. 1-sample $z$-test for proportions $p$ (1- and 2-sided)

# In[63]:


def DS_1sample_ztest_props(y, popmean, k=1, alternative ='two-sided', alpha=0.05):
    """
    *
    Function DS_1sample_ztest_props(y, popmean, k=1, alternative ='two-sided', alpha=0.05)
     
       This function performs a 1-sample z-test (Null Hypothesis Significance Test)
       in the spirit of R, testing 1 proportion using a normal approximation, 
       with or without the continuity correction, and the exact binomial calculation,
       assuming a Binomial(k, p)-distribution. For Bernoulli data, set k = 1 (default).
       The function also evaluates the effect size (Cramer's V2, Cohen's h).
    
    Requires:          scipy.stats.norm, scipy.stats.binom
    
    Usage:             DS_1sample_ztest_props(y, popmean = p*, k, 
                            alternative=['two-sided']/'less'/'greater', alpha = 0.05)
     
                         Note 1: y is an array with Binomial(k, p) data
                         Note 2: If y is an array with BINARY data (0, 1), 
                                 set k = 1 (Bernoulli data)
     
                         k:       Binomial(k, p) parameter = number of Bernoulli repetitions
                         alternative = 'two-sided' [default]  H1: p != p*
                                       'less'                 H1: p < p*
                                       'greater'              H1: p > p*
                         alpha:   significance level of test [default: 0.05]
     
    Return:            z, p-value, z.crit.L, z.crit.R  [ + print interpretable output to stdout ]
                       where z.crit.L and z.crit.R are the lower and upper critical values, 
                       z is the test statistic and p-value is the p-value of the test.    
     
    Author:            M.E.F. Apol
    Date:              2022-01-28, rev. 2022_08_26, 2024_01_02
    Validation:
    """
    
    import numpy as np
    from scipy.stats import norm
    from scipy.stats import binom
    
    n = len(y)
    p_ML = np.mean(y)/k
    p_star = popmean
    N = k * n
    O_1 = np.sum(y)
    O_0 = N - O_1   
    
    print(80*'-')
    print('1-sample z-test for 1 proportion:')
    if k == 1:
        print('     assuming Bernoulli(p) data for dataset')
    else:
        print('     assuming Binomial(' + str(k) + ', p) data for dataset')
    print('Observed dataset: O.1 = {:d}, O.0 = {:d}, N = {:d}'.format(O_1, O_0, N))
    print('p.ML = {:.3g}, p* = {:.3g}, alpha = {:.3g}'.format(p_ML, popmean, alpha))
    print('H0: p  = p*')
    
    if alternative == 'two-sided':
        print('H1: p != p*')
        # Assuming normal approximation (no continuity correction):
        z = (p_ML - popmean)/np.sqrt(p_star*(1-p_star)/N)
        p_value_z = 2 * norm.cdf(-np.abs(z), 0, 1)
        # Assuming normal approximation with continuity correction:
        z_c = (np.abs(p_ML - p_star) - 1/(2*N))/np.sqrt(p_star*(1-p_star)/N)
        p_value_zc = 2*norm.cdf(-np.abs(z_c), 0, 1)
        # General:
        z_crit_L = norm.ppf(alpha/2, 0, 1)
        z_crit_R = norm.ppf(1-alpha/2, 0, 1)
        # Exact Binomial calculation:
        O = p_ML*N
        E = p_star*N
        Delta = np.abs(O - E)
        O_L = E - Delta
        O_R = E + Delta
        p_value_Bin = binom.cdf(O_L, N, p_star) + binom.sf(O_R-1, N, p_star)  
        # subtract 1 from Observed_R, because of discrete distribution
    elif alternative == 'less':
        print('H1: p  < p*')
        # Assuming normal approximation (no continuity correction):
        z = (p_ML - popmean)/np.sqrt(p_star*(1-p_star)/N)
        p_value_z = norm.cdf(z)
        # Assuming normal approximation with continuity correction:
        z_c = (p_ML - p_star + 1/(2*N))/np.sqrt(p_star*(1-p_star)/N)
        p_value_zc = norm.cdf(z_c, 0, 1)
        # General:
        z_crit_L = norm.ppf(alpha, 0, 1)
        z_crit_R = float('inf')
        # Exact Binomial calculation:
        O = p_ML*N
        p_value_Bin = binom.cdf(O, N, p_star)
    elif alternative == 'greater':
        print('H1: p  > p*')
        # Assuming normal approximation (no continuity correction):
        z = (p_ML - popmean)/np.sqrt(p_star*(1-p_star)/N)
        p_value_z = 1 - norm.cdf(z, 0, 1)
        # better precision, use the survival function:
        p_value_z = norm.sf(z, 0, 1)
        # Assuming normal approximation with continuity correction:
        z_c = (p_ML - p_star - 1/(2*N))/np.sqrt(p_star*(1-p_star)/N)
        p_value_zc = norm.sf(z_c, 0, 1)  # better accuracy!
        # General:
        z_crit_L = float('-inf')
        z_crit_R = norm.ppf(1-alpha, 0, 1)
        # Exact Binomial calculation:
        O = p_ML*N
        p_value_Bin = binom.sf(O-1, N, p_star)  # subtract 1 from Observed, 
                                        # because of discrete distribution
    else:
        print('Wrong alternative hypothesis chosen!')
        print(80*'-' + '\n')
        z, p_value_z, z_crit_L, z_crit_R = np.nan, np.nan, np.nan, np.nan
        return(z, p_value_z, z_crit_L, z_crit_R)
    
    # Effect size (Cramer's V2):
    V2 = z**2 / N
    # Effect size (Cohen's h, see Cohen 1988, pp. 181-185):
    h = 2*np.arcsin(np.sqrt(p_ML)) - 2*np.arcsin(np.sqrt(p_star))
    
    print('* Normal approximation:')
    print('z   = {:.4g}, p-value = {:.4g}, z.crit.L = {:.4g}, z.crit.R = {:.4g}'.format(z, p_value_z, z_crit_L, z_crit_R))
    print('* Normal approximation with continuity correction:')
    print('z.c = {:.4g}, p-value = {:.4g}, z.crit.L = {:.4g}, z.crit.R = {:.4g}'.format(z_c, p_value_zc, z_crit_L, z_crit_R))
    print('* Exact Binomial calculation:')
    print('p-value = {:.4g}'.format(p_value_Bin))
    print(80*'.')
    print('Effect size: Cramer\'s V2 = {:.3g}; benchmarks V2: 0.01 = small, 0.09 = medium, 0.25 = large'.format(V2))
    print('Effect size: Cohens\' h = {:.3g}; benchmarks |h|:  0.2 = small, 0.5 = medium, 0.8 = large'.format(h))
    print(80*'-' + '\n')
    return(z, p_value_z, z_crit_L, z_crit_R)


# ## 18. 2-sample $z$-test for proportions (1- and 2-sided)

# In[67]:


def DS_2sample_ztest_props(y1, y2, k1=1, k2=1, alternative='two-sided', alpha=0.05):
    """
    *
    Function DS_2sample_ztest_props(y1, y2, k1=1, k2=1, alternative='two-sided', alpha=0.05)
     
       This function performs a 2-sample z-test (Null Hypothesis Significance Test) 
       in the spirit of R, testing 2 proportions using a normal approximation. 
       We assume that datset y1 ~ Bin(k1, p1) and y2 ~ Bin(k2, p2).
       If we assume that y1 and y2 are binary count data, so y1 ~ Ber(p) and y2 ~ Ber(p), 
       then set k1 = k2 = 1 (default).
       The function also evaluates the effect size (Cramer's V2, Cohen's h).
        
    Requires:          -
     
    Usage:             DS_2sample_ztest_props(y1, y2, k1, k2, 
                            alternative=['two-sided']/'less'/'greater', alpha = 0.05)
     
                         Note 1: y.i (i=1, 2) is an array with Binomial(k.i, p.i) data
                         Note 2: If y.i is an array with BINARY data (0, 1), 
                                 set k.i = 1 (Bernoulli data)
     
                         k1, k2:    Binomial(k, p) parameter = number of Bernoulli repetitions 
                                    of datasets y1 and y2
     
                         alternative = 'two-sided' [default]  H1: p.1 != p.1
                                       'less'                 H1: p.1 <  p.2
                                       'greater'              H1: p.1 >  p.2
                         alpha:   significance level of test [default: 0.05]
     
    Return:            z, p-value, z.crit.L, z.crit.R  [ + print interpretable output to stdout ]
                       where z.crit.L and z.crit.R are the lower and upper critical values, 
                       z is the test statistic and p-value is the p-value of the test.    
      
    Author:            M.E.F. Apol
    Date:              2022-01-29, rev. 2022_08_26, 2024_01_02
    Validation:
    """
    
    import numpy as np
    from scipy.stats import norm
    
    n_1 = len(y1)
    n_2 = len(y2)
    p_ML_1 = np.mean(y1)/k1
    p_ML_2 = np.mean(y2)/k2
    O_1 = np.sum(y1)
    O_2 = np.sum(y2)
    N_1 = k1 * n_1
    N_2 = k2 * n_2
    N = N_1 + N_2
    p_star = (O_1 + O_2)/(N_1 + N_2)
    z = (p_ML_1 - p_ML_2)/np.sqrt(p_star*(1-p_star)*(1/N_1 + 1/N_2))
    
    print(80*'-')
    print('2-sample z-test for 2 proportions:')
    if k1 == 1:
        print('     assuming Bernoulli(p) data for dataset 1')
    else:
        print('     assuming Binomial(' + str(k1) + ', p) data for dataset 1')
    if k2 == 1:
        print('     assuming Bernoulli(p) data for dataset 2')
    else:
        print('     assuming Binomial(' + str(k2) + ', p) data for dataset 2')
    print('Observed dataset 1: O.11 = {:d}, O.10 = {:d}, N.1 = {:d}'.format(O_1, N_1-O_1, N_1))
    print('Observed dataset 2: O.21 = {:d}, O.20 = {:d}, N.2 = {:d}'.format(O_2, N_2-O_2, N_2))
    print('p.ML.1 = {:.3g}, p.ML.2 = {:.3g}, p* = {:.3g}, alpha = {:.3g}'.format(p_ML_1, p_ML_2, p_star, alpha))
    print('H0: p.1  = p.2')
    
    if alternative == 'two-sided':
        print('H1: p.1 != p.2')
        p_value = 2 * norm.cdf(-np.abs(z), 0, 1)
        z_crit_L = norm.ppf(alpha/2, 0, 1)
        z_crit_R = norm.ppf(1-alpha/2, 0, 1)
    elif alternative == 'less':
        print('H1: p.1  < p.2')
        p_value = norm.cdf(z)
        z_crit_L = norm.ppf(alpha, 0, 1)
        z_crit_R = float('inf')
    elif alternative == 'greater':
        print('H1: p.1  > p.2')
        p_value = 1 - norm.cdf(z, 0, 1)
        # better precision, use the survival function:
        p_value = norm.sf(z, 0, 1)
        z_crit_L = float('-inf')
        z_crit_R = norm.ppf(1-alpha, 0, 1)
    else:
        print('Wrong alternative hypothesis chosen!')
        print(80*'-' + '\n')
        z, p_value, z_crit_L, z_crit_R = np.nan, np.nan, np.nan, np.nan
        return(z, p_value, z_crit_L, z_crit_R)
    
    # Effect size (Cramer's V2):
    V2 = z**2 / N
    # Effect size (Cohen's h):
    h = 2*np.arcsin(np.sqrt(p_ML_1)) - 2*np.arcsin(np.sqrt(p_ML_2))
    
    print('z = {:.4g}, p-value = {:.4g}, z.crit.L = {:.4g}, z.crit.R = {:.4g}'.format(z, p_value, z_crit_L, z_crit_R))
    print(80*'.')
    print('Effect size: Cramers V2 = {:.3g}; benchmarks V2: 0.01 = small, 0.09 = medium, 0.25 = large'.format(V2))
    print('Effect size: Cohens h = {:.3g}; benchmarks |h|: 0.2 = small, 0.5 = medium, 0.8 = large'.format(h))
    print(80*'-' + '\n')
    return(z, p_value, z_crit_L, z_crit_R)


# ## 19. McNemar $\chi^2$-test for 2 paired proportions

# In[71]:


def DS_paired_McNemar_chi2_test_props(y1, y2, alternative='two-sided', alpha=0.05):
    """
    *
    Function DS_paired_McNemar_chi2_test_props(y1, y2, alternative='two-sided', alpha=0.05)
    
       This function tests two proportions p.1 and p.2 of paired data y1 and y2 (Null Hypothesis Significance Test).
       The data is assumed to be Binomial, with 0 = failure and 1 = succes; 
       p.1 and p.2 are th        e proprtions of success for raters 1 and 2, respectively.
       
    
    Requires:            scipy.stats.norm, scipy.stats.binom, scipy.stats.chi2
    
    Usage:               DS_paired_McNemar_chi2_test_props(y1, y2, 
                              alternative=['two-sided']/'less'/'greater', alpha=0.05)
    
    Arguments:
      y1, y2             data arrays
      alternative        'two-sided' [default]   H1: n.12 != n.21  [or: p.1 != p.2]
                         'less'                  H1: n.12 <  n.21  [or: p.1  < p.2]
                         'greater'               H1: n.12 >  n.21  [or: p.1  > p.2]
      alpha              significance level of test [default: 0.05]     
 
 
    Returns:             z, p_value [ + print interpretable output to stdout ]
    where
      z                  z-score statistic of normal approximation
      p_value            p-value of z-score statistic
      
    Validation:          against Minitab 21 (2024-01-02), R (prop.test and binom.test)
      
    Author:              M.E.F. Apol
    Date:                2024-01-03
    """    
    
    
    
    import numpy as np
    from scipy.stats import chi2
    from scipy.stats import norm
    from scipy.stats import binom
    
    n_1 = len(y1); n_2 = len(y2)
    
    if n_1 != n_2:
        raise ValueError('Both datasets must be same size...')
    
    
    def DS_xtab(*cols, apply_wt=False):
    
    ################################################################################
    #
    # Define a function to generate a (1-way or) 2-way contingency table from
    # (a single or) two arrays of equal length.
    #   assuming a normal approximation
    #
    # Author:            Doug Ybarbo, adaptation M.E.F. Apol
    # Source:            https://gist.github.com/alexland/d6d64d3f634895b9dc8e
    # Date:              2021-11-30
    # Usage:             my_xtab(y1 [, y2]) 
    #
    #                    Note: y is an array with BINARY data (0, 1)
    #
    # Return:            uv, table
    #                    where
    #                       uv = tuple containing the row and column headers
    #                       table = contingency table
    #
    ################################################################################
  
    
        if not all(len(col) == len(cols[0]) for col in cols[1:]):
            raise ValueError("all arguments must be same size")

        if len(cols) == 0:
            raise TypeError("xtab() requires at least one argument")

        fnx1 = lambda q: len(q.squeeze().shape)
        if not all([fnx1(col) == 1 for col in cols]):
            raise ValueError("all input arrays must be 1D")
    
        if apply_wt:
            cols, wt = cols[:-1], cols[-1]
        else:
            wt = 1

        uniq_vals_all_cols, idx = zip( *(np.unique(col, return_inverse=True) for col in cols) )
        shape_xt = [uniq_vals_col.size for uniq_vals_col in uniq_vals_all_cols]
        dtype_xt = 'float' if apply_wt else 'uint'
        xt = np.zeros(shape_xt, dtype=dtype_xt)
        np.add.at(xt, idx, wt)
        return uniq_vals_all_cols, xt
    
    ######### end DS_xtab function
    
    _, XXX = DS_xtab(y_1, y_2)
    
    # assign matrix elements:
    n_11, n_12, n_21, n_22 = XXX[1,1], XXX[1,0], XXX[0,1], XXX[0,0]
    # sum of off-diagonal elements:
    n = n_12 + n_21
    n_min = np.min([n_12, n_21])
    # total number of pairs:
    n_tot = n_11 + n_12 + n_21 + n_22
    # marginal totals:
    n_1s = n_11 + n_12
    n_2s = n_21 + n_22
    n_s1 = n_11 + n_21
    n_s2 = n_12 + n_22
    
    # Proportions of "success" (i.e., y = 1):
    p_rater1 = n_1s / n_tot  # proportion succes rater 1
    p_rater2 = n_s1 / n_tot  # proportion succes rater 2
    
    
    print(80*'-')
    print('Repeated measures McNemar chi2-test for 2 paired proportions:')
    print('     assuming Bernoulli(p) data for both datasets')
    print('Proportions of succes: p (rater 1) = {:.3g}, p (rater 2) = {:.3g}, alpha = {:.3g}'.format(p_rater1, p_rater2, alpha))
    print()
    print('Repeated measures cross tabs:')
    print('\t\t\t|{:^15s}|'.format('Rater 2'))
    print('\t\t\t' + ' ' + 15*'-')
    print('\t\t\t|{:^6d}\t|{:^6d}\t|{:^7s}'.format(1,0, 'Total'))
    print(24*'-' + 24*'-')
    print('{:^15s}\t|{:^6d}\t|{:^6d}\t|{:^6d}\t|{:^6d}'.format('Rater 1', 1, n_11, n_12, n_11+n_12))
    print('\t\t|{:^6d}\t|{:^6d}\t|{:^6d}\t|{:^6d}'.format(0, n_21, n_22, n_21+n_22))
    print(24*'-' + 24*'-')
    print('{:^23s}\t|{:^6d}\t|{:^6d}\t|{:^6d}'.format('Total', n_11+n_21, n_12+n_22, n_11+n_12+n_21+n_22))
    print()
    print('n.12 = {:d}, n.21 = {:d}'.format(n_12, n_21))
    print()
    
    print('H0: n.12  = n.21 [or: p (rater 1)  = p (rater 2)]')
    
    if alternative == 'two-sided':
        print('H1: n.12 != n.21 [or: p (rater 1) != p (rater 2)]')
        
        ###
        # McNemar chi2-test:
        chi2_MN = (n_12 +-1* n_21)**2 / (n_12+n_21)
        chi2_crit = chi2.ppf(0.95, 1)
        p_value_MN = 1 - chi2.cdf(chi2_MN, 1)
        # Alternative: uncorrected 1-sample z-test for proportions:
        p_star = 0.50
        p_hat = n_12/n
        z_MNz = (p_hat - p_star)/np.sqrt(p_star*(1-p_star)/n)
        z_crit_L, z_crit_R = norm.ppf([0.025, 0.975], 0, 1)
        p_value_MNz = 2 * norm.cdf(-1*np.abs(z_MNz), 0, 1)
        ###
        # McNemar chi2-test with continuity correction:
        chi2_MNc = (np.abs(n_12 +-1* n_21)-1)**2 / (n_12+n_21)
        chi2_critc = chi2.ppf(0.95, 1)
        p_value_MNc = 1 - chi2.cdf(chi2_MNc, 1)
        # Alternative: uncorrected 1-sample z-test for proportions:
        p_star = 0.50
        p_hat = n_12/n
        z_MNzc = (np.abs(p_hat - p_star)-1/(2*n))/np.sqrt(p_star*(1-p_star)/n)
        z_crit_L, z_crit_R = norm.ppf([0.025, 0.975], 0, 1)
        p_value_MNzc = 2 * norm.cdf(-1*np.abs(z_MNzc), 0, 1)
        ###
        # Binomial test:
        n_min = np.min([n_12, n_21])
        p_hat = n_min/n
        p_star = 0.50
        p_value_bin = 2*binom.cdf(n_min, n, 0.5)
        
    elif alternative == 'less':
        print('H1: n.12  = n.21 [or: p (rater 1)  < p (rater 2)]')
        
        #
        # McNemar chi2-test: uncorrected 1-sample z-test for proportions:
        p_star = 0.50
        p_hat = n_min/n
        z_MNz = (p_hat - p_star)/np.sqrt(p_star*(1-p_star)/n)
        z_crit_L = float('-inf')
        z_crit_R = norm.ppf(0.95, 0, 1)
        p_value_MNz = norm.cdf(z_MNz, 0, 1)
        #
        # McNemar chi2-test: 1-sample z-test for proportions with continuity correction:
        p_star = 0.50
        p_hat = n_min/n
        z_MNzc = (p_hat - p_star + 1/(2*n))/np.sqrt(p_star*(1-p_star)/n)
        z_crit_L = float('-inf')
        z_crit_R = norm.ppf(0.95, 0, 1)
        p_value_MNzc = norm.cdf(z_MNzc, 0, 1)
        #
        # Exact Binomial test:
        p_hat = n_12/n
        p_star = 0.50
        p_value_bin = binom.cdf(n_12, n, 0.5)
        
        
    elif alternative == 'greater':
        print('H1: n.12  = n.21 [or: p (rater 1)  > p (rater 2)]')
        
        #
        # McNemar chi2-test: uncorrected 1-sample z-test for proportions:
        p_star = 0.50
        p_hat = n_min/n
        z_MNz = (p_hat - p_star)/np.sqrt(p_star*(1-p_star)/n)
        z_crit_L = float('-inf')
        z_crit_R = norm.ppf(0.95, 0, 1)
        p_value_MNz = norm.sf(z_MNz, 0, 1) # use survival function instead of 1 - cdf for better precision!
        #
        # McNemar chi2-test: 1-sample z-test for proportions with continuity correction:
        p_star = 0.50
        p_hat = n_min/n
        z_MNzc = (p_hat - p_star - 1/(2*n))/np.sqrt(p_star*(1-p_star)/n)
        z_crit_L = float('-inf')
        z_crit_R = norm.ppf(0.95, 0, 1)
        p_value_MNzc = norm.sf(z_MNzc, 0, 1)
        #
        # Exact Binomial test:
        p_hat = n_12/n
        p_star = 0.50
        p_value_bin = binom.sf(n_12 - 1, n, 0.5)  # subtract 1 because for discrete P(Y >= x)
        
    else:
        raise ValueError('Wrong alternative hypothesis chosen!')
    
    ### Effect Sizes and additional statistics:
    # Effect sizes: proportion of overall agreement
    p_overall = (n_11 + n_22)/n_tot  # Uebersax, 2009
    s_p_overall = np.sqrt(p_overall*(1-p_overall)/n_tot)  # Wald
    CI_p_overall = norm.ppf([0.025, 0.975], p_overall, s_p_overall)
    # Effect size: Cohen's kappa
    p_err = (n_1s*n_s1 + n_2s*n_s2) / (n_tot**2)  # proprtion of agreement due to chance, Cohen 1960
    kappa = (p_overall - p_err)/(1 - p_err)
    s_kappa = np.sqrt(p_overall*(1-p_overall)/(n_tot*(1-p_err)**2))
    CI_kappa = norm.ppf([0.025, 0.975], kappa, s_kappa)
    # Effect size: proportions of specific agreement, Mackinnon, 2000
    p_1 = 2*n_11/(n_1s + n_s1)  # agreement on y = 1
    s_p_1 = np.sqrt(4*n_11*(n_1s+n_s1-2*n_11)*(n_1s+n_s1-1*n_11))/(n_1s+n_s1)**2
    CI_p_1 = norm.ppf([0.025, 0.975], p_1, s_p_1)
    p_0 = 2*n_22/(n_2s + n_s2)  # agreement on y = 0
    s_p_0 = np.sqrt(4*n_22*(n_2s+n_s2-2*n_22)*(n_2s+n_s2-1*n_22))/(n_2s+n_s2)**2
    CI_p_0 = norm.ppf([0.025, 0.975], p_0, s_p_0)
    # Sensitivity (=% true positives), assuming rater 2 = Golden Standard
    SN = n_11/n_s1
    s_SN = np.sqrt(SN*(1-SN)/n_s1)
    CI_SN = norm.ppf([0.025, 0.975], SN, s_SN)
    # Specificity (=% true negatives), assuming rater 2 = Golden Standard
    SP = n_22/n_s2
    s_SP = np.sqrt(SP*(1-SP)/n_s2)
    CI_SP = norm.ppf([0.025, 0.975], SP, s_SP)
    
    
    
    print('* McNemar chi2-test:')
    if alternative == 'two-sided':
        print('chi2.MN = {:.3g}, chi2.crit = {:.3g}, p-value = {:.3g}'.format(chi2_MN, chi2_crit, p_value_MN))
    print('z.MN = {:.3g}, z.crit.L = {:.3g}, z.crit.R = {:.3g}, p-value = {:.3g}'.format(z_MNz, z_crit_L, z_crit_R, p_value_MNz))
    print('* McNemar chi2-test with continuity correction:')
    if alternative == 'two-sided':
        print('chi2.MNc = {:.3g}, chi2.crit = {:.3g}, p-value = {:.3g}'.format(chi2_MNc, chi2_critc, p_value_MNc))
    print('z.MNc = {:.3g}, z.crit.L = {:.3g}, z.crit.R = {:.3g}, p-value = {:.3g}'.format(z_MNzc, z_crit_L, z_crit_R, p_value_MNzc))
    print('* Exact Binomial test:')
    print('p.hat = {:.3g}, p* = 0.50, p-value: {:.3g}'.format(p_hat, p_value_bin))
    print(80*'.')
    print('* Effect size:             Cohen\'s kappa = {:.3f}, CI = [{:.3f}, {:.3f}]; '.format(kappa, CI_kappa[0], CI_kappa[1]))
    print('          benchmarks kappa: 0.1 = small, 0.3 = medium, 0.5 = large')
    print('Proportion of overall agreement:   p.o   = {:.3f}, CI = [{:.3f}, {:.3f}]'.format(p_overall, CI_p_overall[0], CI_p_overall[1]))
    print('Proportions of specific agreement: p.(1) = {:.3f}, CI = [{:.3f}, {:.3f}]]'.format(p_1, CI_p_1[0], CI_p_1[1]))
    print('Proportions of specific agreement: p.(0) = {:.3f}, CI = [{:.3f}, {:.3f}]]'.format(p_0, CI_p_0[0], CI_p_0[1]))
    print('Assuming Rater2 = Golden Standard method:')
    print('Sensitivity (% true positives):       SN = {:.3f}, CI = [{:.3f}, {:.3f}]'.format(SN, CI_SN[0], CI_SN[1]))
    print('Specificity (% true negatives):       SP = {:.3f}, CI = [{:.3f}, {:.3f}]'.format(SP, CI_SP[0], CI_SP[1]))
    print(80*'-')
    return;
    


# ## <span style="color:blue">*Tests for rates or counts (assuming a Poisson distribution):*</span>

# ## 20. 1-sample $z$-test for Poisson counts $\lambda$ (1-and 2-sided)

# In[74]:


def DS_1sample_ztest_counts(y, popmean, alternative ='two-sided', alpha=0.05, method='normal'):
    """
    *
    Function DS_1sample_ztest_counts(y, popmean, alternative ='two-sided', alpha=0.05)
     
       This function performs a 1-sample z-test (Null Hypothesis Significance Test)
       in the spirit of R, testing 1 lambda parameter using a normal approximation, 
       assuming a Poisson(lambda)-distribution.
       [TO DO: The function also evaluates the effect size (Cramer's V2).]
    
    Requires:          numpy, scipy.stats.norm, scipy.stats.poisson
    
    Usage:             DS_1sample_ztest_counts(y, popmean = lambda*,  
                            alternative=['two-sided']/'less'/'greater', alpha = 0.05, 
                            method=['normal']/'Poisson')
     
                         alternative = 'two-sided' [default]  H1: lambda != lambda*
                                       'less'                 H1: lambda < lambda*
                                       'greater'              H1: lambda > lambda*
                         alpha:   significance level of test [default: 0.05]
                         method      = 'normal' [default]     return p-value with normal approximation
                                       'Poisson'              return p-value using Poisson expression
      
    Return:            z, p-value, z.crit.L, z.crit.R  [ + print interpretable output to stdout ]
                       where z.crit.L and z.crit.R are the lower and upper critical values, 
                       z is the test statistic and p-value is the p-value of the test (see 'method)'.    
     
    Author:            M.E.F. Apol
    Date:              2022-12-12
    Validation:
    """
    
    import numpy as np
    from scipy.stats import norm
    from scipy.stats import poisson
    
    n = len(y)
    lambda_ML = np.mean(y)
    lambda_star = popmean
    # Normal approximation:
    z = (lambda_ML - lambda_star)/np.sqrt(lambda_star/n)
    # Poisson expression:
    Lambda = n*lambda_ML
    Lambda_star = n*lambda_star
    Delta = n*np.abs(lambda_ML - lambda_star)
    Lambda_L = Lambda_star - Delta
    Lambda_U = Lambda_star + Delta
    
    print(80*'-')
    print('1-sample z-test for 1 lambda:')
    print('     assuming Poisson(lambda) data for dataset')
    print('lambda.ML = {:.3g}, lambda* = {:.3g}, n = {:d}, alpha = {:.3g}'.format(lambda_ML, lambda_star, n, alpha))
    print('H0: lambda  = lambda*')
    
    if alternative == 'two-sided':
        print('H1: lambda != lambda*')
        # Normal approximation:
        p_value_N = 2 * norm.cdf(-np.abs(z), 0, 1)
        # Poisson expression:
        p_value_P = poisson.cdf(Lambda_L, Lambda_star) + poisson.sf(Lambda_U, Lambda_star) + poisson.pmf(Lambda_U, Lambda_star)
        z_crit_L = norm.ppf(alpha/2, 0, 1)
        z_crit_R = norm.ppf(1-alpha/2, 0, 1)
    elif alternative == 'less':
        print('H1: lambda  < lambda*')
        # Normal approximation:
        p_value_N = norm.cdf(z)
        # Poisson expression:
        p_value_P = poisson.cdf(Lambda, Lambda_star)
        z_crit_L = norm.ppf(alpha, 0, 1)
        z_crit_R = float('inf')
    elif alternative == 'greater':
        print('H1: lambda  > lambda*')
        # Normal approxumation:
        p_value_N = 1 - norm.cdf(z, 0, 1)
        # better precision, use the survival function:
        p_value_N = norm.sf(z, 0, 1)
        # Poisson expression:
        p_value_P = poisson.sf(Lambda, Lambda_star) + poisson.pmf(Lambda, Lambda_star)
        z_crit_L = float('-inf')
        z_crit_R = norm.ppf(1-alpha, 0, 1)
    else:
        print('Wrong alternative hypothesis chosen!')
        print(80*'-' + '\n')
        z, p_value, z_crit_L, z_crit_R = np.nan, np.nan, np.nan, np.nan
        return(z, p_value_N, z_crit_L, z_crit_R)
    
    # Effect size (Cramer's V2):   # Tentative...
    V2 = z**2 / n
    # Choose the p-value according to 'method':
    if method == 'norm':
        p_value = p_value_N
    else:
        p_value = p_value_P
    
    # To fix later:
    #print('z = {:.4g}, p-value = {:.4g} (normal), p-value = {:.4g} (Poisson), z.crit.L = {:.4g}, z.crit.R = {:.4g}'.format(z, p_value_N, p_value_P, z_crit_L, z_crit_R))
    print('z = {:.4g}, p-value = {:.4g}, z.crit.L = {:.4g}, z.crit.R = {:.4g}'.format(z, p_value_N, z_crit_L, z_crit_R))
    #print(80*'.')
    #print('Effect size: V2 = {:.3g}; benchmarks V2: 0.01 = small, 0.09 = medium, 0.25 = large (???)'.format(V2))
    print(80*'-' + '\n')
    return(z, p_value, z_crit_L, z_crit_R)


# ## 21. 2-sample $z$-test for Poisson counts $\lambda$ (1-and 2-sided)

# In[79]:


def DS_2sample_ztest_counts(y1, y2, alternative ='two-sided', alpha=0.05, method='normal'):
    """
    *
    Function DS_2sample_ztest_counts(y1, y2, alternative ='two-sided', alpha=0.05)
     
       This function performs a 2-sample z-test (Null Hypothesis Significance Test)
       in the spirit of R, testing 2 lambda parameters using a normal approximation, 
       assuming both datasets come from Poisson(lambda.1) and Poisson(lambda.2)-distributions.
       [TO DO: The function also evaluates the effect size (Cramer's V2).]
    
    Requires:          -
    
    Usage:             DS_2sample_ztest_counts(y1, y2,  
                            alternative=['two-sided']/'less'/'greater', alpha = 0.05, 
                            method=['normal'])
     
                         alternative = 'two-sided' [default]  H1: lambda.1 != lambda.2
                                       'less'                 H1: lambda.1 < lambda.2
                                       'greater'              H1: lambda.1 > lambda.2
                         alpha:   significance level of test [default: 0.05]
                         method      = 'normal' [default]     return p-value with normal approximation
                                       
      
    Return:            z, p-value, z.crit.L, z.crit.R  [ + print interpretable output to stdout ]
                       where z.crit.L and z.crit.R are the lower and upper critical values, 
                       z is the test statistic and p-value is the p-value of the test (see 'method)'.    
     
    Author:            M.E.F. Apol
    Date:              2022-12-12
    Validation:
    """
    
    import numpy as np
    from scipy.stats import norm
    from scipy.stats import poisson
    
    n_1 = len(y1) ; n_2 = len(y2)
    lambda_1_ML = np.mean(y1); lambda_2_ML = np.mean(y2)
    # Normal approximation:
    lambda_p = (n_1*lambda_1_ML + n_2*lambda_2_ML)/(n_1 + n_2) # pooled value
    z = (lambda_1_ML - lambda_2_ML)/np.sqrt( lambda_p * (1/n_1 + 1/n_2) )
    
    print(80*'-')
    print('2-sample z-test for lambda:')
    print('     assuming Poisson(lambda.1) data for dataset 1')
    print('     assuming Poisson(lambda.2) data for dataset 2')
    print('lambda.1.ML = {:.3g}, lambda.2.ML = {:.3g}, n.1 = {:d}, n.2 = {:d}, alpha = {:.3g}'.format(lambda_1_ML, lambda_2_ML, n_1, n_2, alpha))
    print('H0: lambda.1  = lambda.2')
    
    if alternative == 'two-sided':
        print('H1: lambda.1 != lambda.2')
        # Normal approximation:
        p_value_N = 2 * norm.cdf(-np.abs(z), 0, 1)
        z_crit_L = norm.ppf(alpha/2, 0, 1)
        z_crit_R = norm.ppf(1-alpha/2, 0, 1)
    elif alternative == 'less':
        print('H1: lambda.1  < lambda.2')
        # Normal approximation:
        p_value_N = norm.cdf(z)
        z_crit_L = norm.ppf(alpha, 0, 1)
        z_crit_R = float('inf')
    elif alternative == 'greater':
        print('H1: lambda.1  > lambda.2')
        # Normal approxumation:
        p_value_N = 1 - norm.cdf(z, 0, 1)
        # better precision, use the survival function:
        p_value_N = norm.sf(z, 0, 1)
        z_crit_L = float('-inf')
        z_crit_R = norm.ppf(1-alpha, 0, 1)
    else:
        print('Wrong alternative hypothesis chosen!')
        print(80*'-' + '\n')
        z, p_value, z_crit_L, z_crit_R = np.nan, np.nan, np.nan, np.nan
        return(z, p_value_N, z_crit_L, z_crit_R)
    
    # Effect size (Cramer's V2):   # Tentative...
    V2 = z**2 * (1/n_1 + 1/n_2)
    # Choose the p-value according to 'method':
    if method == 'norm':
        p_value = p_value_N
    else:
        p_value = p_value_N
    
    # To fix later:
    #print('z = {:.4g}, p-value = {:.4g} (normal), z.crit.L = {:.4g}, z.crit.R = {:.4g}'.format(z, p_value_N, z_crit_L, z_crit_R))
    print('z = {:.4g}, p-value = {:.4g}, z.crit.L = {:.4g}, z.crit.R = {:.4g}'.format(z, p_value_N, z_crit_L, z_crit_R))
    #print(80*'.')
    #print('Effect size: V2 = {:.3g}; benchmarks V2: 0.01 = small, 0.09 = medium, 0.25 = large (???)'.format(V2))
    print(80*'-' + '\n')
    return(z, p_value, z_crit_L, z_crit_R)


# ## <span style="color:blue">*Analysis of calibration curves (linear regression):*</span>

# ## 22.  Analysis of usual polynomial calibration models

# We have response values $y$ for given independent factor (feature) values $x$:
# 
# `calibration = DS_CalibrationAnalysis(x, y)`
# 
# **Step 1**: Perform regression on the following list of nested (polynomial) regression models:
# 
# - **Model 0**: $y = a_0$
# 
# - **Model 1**: $y = a_0 + a_1 \cdot x$
# 
# - **Model 1a**: $y = a_1 \cdot x$
# 
# - **Model 2**: $y = a_0 + a_1 \cdot x + a_2 \cdot x^2$
# 
# - **Model 2a**: $y = a_1 \cdot x + a_2 \cdot x^2$
# 
# - **Model 2b**: $y = a_2 \cdot x^2$
# 
# - **Model 2c**: $y = a_0 + a_2 \cdot x^2$
# 
# `calibration.fit()`
# 
# **Step 2**: Select the "best" calibration model based on the (small sample) Akaike Information Criterion (AIC.c).
# 
# `calibration.summary()`
# 
# **Step 3a**: Obtain detailed regression statistics for the "best" model.
# 
# `calibration.results_model(model).summary()`
# 
# **Step 3b**: Plot the "best" calibration model with CI together with experimental data.
# 
# `calibration.plot_model(model)`
# 
# **Step 4**: Predict using the "best" model the response value $y_p$ with CI for given $x_p$.
# 
# `calibration.predict(xp, model, confidence)`
# 
# **Step 5**: Interpolate using the "best" model the $x_0$ value with CI for $m$ replica measurements $y_0$.
# 
# `calibration.interpolate(y_0, model, confidence)`

# In[84]:


class DS_CalibrationAnalysis:
    
    def __init__(self, x, y, bVerbatim=True):
        """
        *
        Function DS_CalibrationAnalysis(x, y)
         
           Analyze all 2nd order polynomial calibration models one by one, and collect relevant info.
          
           Models are:                                   model_id:
               Model 0:   y = a.0                        "0"
               Model 1:   y = a.0 + a.1*x                "1"
               Model 1a:  y =       a.1*x                "1a"
               Model 2:   y = a.0 + a.1*x + a.2*x^2      "2"
               Model 2a:  y =       a.1*x + a.2*x^2      "2a"
               Model 2b:  y =               a.2*x^2      "2b"
               Model 2c:  y = a.0         + a.2*x^2      "2c"
        
           Input:
               x, y                  independent factor (feature), respons, both 1D-arrays
               bVerbatim             True: produce a scatterplot of the data
               
           Output:
               object with several methods:
               
               .fit()                        fit all models to the data x,y
               .summary()                    give a tabular summary of the various calibration models 
                                             (R2, AIC.c etc.)
               .results_model(model_id)      retrieve the OLS results of model model_id
               .plot_model(model_id, ...)    plot the predictions of model model_id with CI together
                                             with the data; other plotting options are available.
               .predict(x_p, model_id, confidence)
                                             predict at value x_p a new y_p value with CI
               .interpolate(y_0, model_id, confidence)
                                             interpolate the value(s) y_0 back to x_0 value with CI 
                                             
            Requires:     -
            Loads:        numpy, statsmodels.api, scipy.stats, matplotlib.pyplot
               
            Author:       M.Emile F. Apol
            Date:         2022-09-14, revisions 2022-11-29
        """
        
        import numpy as np
        import statsmodels.api as sm
        from scipy.stats import t
        import matplotlib.pyplot as plt
        
        # To be sure:
        x = np.array(x)
        y = np.array(y)
        
        # Make libraries available within class:
        self.np = np
        self.sm = sm
        self.t = t
        self.plt = plt
        
        # Make data available within class:
        self.x = x
        self.y = y
        
        if(bVerbatim):
            # For convenience, produce a preliminary scatter plot:
            plt.scatter(x, y, marker='o', color='black', label='Experimental')
            plt.xlabel('Independent factor, $x$')
            plt.ylabel('Response, $y$')
            plt.legend(loc='best')
            plt.title('Calibration data')
            plt.show()
        
        pass;
    
    
    
    def fit(self, bVerbatim=True):
        """
        *
        Method DS_CalibrationAnalysis.fit()
        
           Fit all calibration models to the x,y data, and collect info.
        
           Input:
              bVerbatim               True : print output of each model to stdout
                                      False: do not print
        
           Author:          M.Emile F. Apol
           Date:            2022-09-21
        """
        
        sm = self.sm
        np = self.np
        x = self.x
        y = self.y
        
        def DS_AIC(fit):
    
        ###########################################################################
        #
        # Calculate the AIC and AIC.c-values of a LS linear regression model
        # based on the 'fit' object from an OLS fit using statsmodels.api
        #
        # Equations:
        #
        # AIC = n*log(SS.err/n) + 2*(P+1)
        # AIC.c = n*log(SS.err/n) + 2*(P+1) + 2*(P+1)*(P+2)/(n-P-2)
        # 
        # with P the number of parameters a.i in the regression model.
        #
        # Author:        M.E.F. Apol
        # Date:          2020-01-06
        #
        # Return:        AIC, AIC.c
        #
        ###########################################################################
    
            SS_err = fit.ssr
            n = fit.nobs
            P = len(fit.params)
            AIC = n * np.log(SS_err/n) + 2 * (P + 1)
            if(n-P-2 > 0):
                AIC_c = AIC + 2 * (P + 1) * (P + 2) / (n - P - 2)
            else:
                AIC_c = np.nan
            return(AIC, AIC_c);
        
        
        ###########################
        # Analyze all calibration models one by one:
        ###########################
        
        Models = ["0","1","1a","2","2a","2b","2c"]
        Formulas = ["y = a.0","y = a.0 + a.1*x", "y = a.1*x", "y = a.0 + a.1*x + a.2*x^2", 
                    "y = a.1*x + a.2*x^2", "y = a.2*x^2", "y = a.0 + a.2*x^2"]
        Mod_d = {
            "0": 0,
            "1": 1,
            "1a": 2,
            "2": 3,
            "2a": 4,
            "2b": 5,
            "2c": 6
        }
        
        if(bVerbatim):
            print()
            print('Analyzing all calibration models:')
            print("-"*80)
        
        ###########################
        # Model 0: y = a.0
        ###########################
        
        n = len(x)
        X_0 = np.ones(n)
        model = sm.OLS(y,X_0)
        results_0 = model.fit()
        a_0_0 = results_0.params[0]; a_1_0 = np.nan; a_2_0 = np.nan
        s_a_0_0 = results_0.bse[0]; s_a_1_0 = np.nan; s_a_2_0 = np.nan
        p_a_0_0 = results_0.pvalues[0]; p_a_1_0 = np.nan; p_a_2_0 = np.nan
        R2_0 = 0.0; R2_adj_0 = 0.0
        AIC_0, AIC_c_0 = DS_AIC(results_0)
        s2_yx_0 = results_0.scale
        V_0 = results_0.cov_params()
        
        if(bVerbatim):
            print('Model ' + Models[Mod_d["0"]] + ": " + Formulas[Mod_d["0"]])
            print('Parameters:  a.0 = {:.3g}, a.1 = {:.3g}, a.2 = {:.3g}'.format(a_0_0, a_1_0, a_2_0))
            print('P-values:    p(a.0) = {:.3g}, p(a.1) = {:.3g}, p(a.2) = {:.3g}'.format(p_a_0_0, p_a_1_0, p_a_2_0))
            print('Effect size: R2 = {:.5f}, R2.adj = {:.5f}'.format(R2_0, R2_adj_0))
            print('AIC = {:.2f}, AIC.c = {:.2f}'.format(AIC_0, AIC_c_0))
            print("-"*80)
        
        ###########################
        # Model 1: y = a.0 + a.1*x
        ###########################
        
        # First, add the constant a_0 to the model:
        X_1 = sm.add_constant(x)
        model = sm.OLS(y,X_1)
        results_1 = model.fit()
        a_0_1 = results_1.params[0]; a_1_1 = results_1.params[1]; a_2_1 = np.nan
        s_a_0_1 = results_1.bse[0]; s_a_1_1 = results_1.bse[1]; s_a_2_1 = np.nan
        p_a_0_1 = results_1.pvalues[0]; p_a_1_1 = results_1.pvalues[1]; p_a_2_1 = np.nan
        R2_1 = results_1.rsquared; R2_adj_1 = results_1.rsquared_adj
        AIC_1, AIC_c_1 = DS_AIC(results_1)
        s2_yx_1 = results_1.scale
        V_1 = results_1.cov_params()
        
        if(bVerbatim):
            print('Model ' + Models[Mod_d["1"]] + ": " + Formulas[Mod_d["1"]])
            print('Parameters:  a.0 = {:.3g}, a.1 = {:.3g}, a.2 = {:.3g}'.format(a_0_1, a_1_1, a_2_1))
            print('P-values:    p(a.0) = {:.3g}, p(a.1) = {:.3g}, p(a.2) = {:.3g}'.format(p_a_0_1, p_a_1_1, p_a_2_1))
            print('Effect size: R2 = {:.5f}, R2.adj = {:.5f}'.format(R2_1, R2_adj_1))
            print('AIC = {:.2f}, AIC.c = {:.2f}'.format(AIC_1, AIC_c_1))
            print("-"*80)
        
        ###########################
        # Model 1a: y = a.1*x
        ###########################
        
        # First, do not add the constant a_0 to the model:
        X_1a = x
        model = sm.OLS(y,X_1a)
        results_1a = model.fit()
        a_0_1a = np.nan; a_1_1a = results_1a.params[0]; a_2_1a = np.nan
        s_a_0_1a = np.nan; s_a_1_1a = results_1a.bse[0]; s_a_2_1a = np.nan
        p_a_0_1a = np.nan; p_a_1_1a = results_1a.pvalues[0]; p_a_2_1a = np.nan
        R2_1a = results_1a.rsquared; R2_adj_1a = results_1a.rsquared_adj
        AIC_1a, AIC_c_1a = DS_AIC(results_1a)
        s2_yx_1a = results_1a.scale
        V_1a = results_1a.cov_params()
        
        if(bVerbatim):
            print('Model ' + Models[Mod_d["1a"]] + ": " + Formulas[Mod_d["1a"]])
            print('Parameters:  a.0 = {:.3g}, a.1 = {:.3g}, a.2 = {:.3g}'.format(a_0_1a, a_1_1a, a_2_1a))
            print('P-values:    p(a.0) = {:.3g}, p(a.1) = {:.3g}, p(a.2) = {:.3g}'.format(p_a_0_1a, p_a_1_1a, p_a_2_1a))
            print('Effect size: R2 = {:.5f}, R2.adj = {:.5f}'.format(R2_1a, R2_adj_1a))
            print('AIC = {:.2f}, AIC.c = {:.2f}'.format(AIC_1a, AIC_c_1a))
            print("-"*80)
            
        ###########################
        # Model 2: y = a.0 + a.1*x + a.2*x^2
        ###########################
        
        # First, add the x^2 column and a constant a_0 to the model:
        X_2 = np.column_stack((x, x**2))
        X_2 = sm.add_constant(X_2)
        model = sm.OLS(y,X_2)
        results_2 = model.fit()
        a_0_2 = results_2.params[0]; a_1_2 = results_2.params[1]; a_2_2 = results_2.params[2]
        s_a_0_2 = results_2.bse[0]; s_a_1_2 = results_2.bse[1]; s_a_2_2 = results_2.bse[2]
        p_a_0_2 = results_2.pvalues[0]; p_a_1_2 = results_2.pvalues[1]; p_a_2_2 = results_2.pvalues[2]
        R2_2 = results_2.rsquared; R2_adj_2 = results_2.rsquared_adj
        AIC_2, AIC_c_2 = DS_AIC(results_2)
        s2_yx_2 = results_2.scale
        V_2 = results_2.cov_params()
        
        if(bVerbatim):
            print('Model ' + Models[Mod_d["2"]] + ": " + Formulas[Mod_d["2"]])
            print('Parameters:  a.0 = {:.3g}, a.1 = {:.3g}, a.2 = {:.3g}'.format(a_0_2, a_1_2, a_2_2))
            print('P-values:    p(a.0) = {:.3g}, p(a.1) = {:.3g}, p(a.2) = {:.3g}'.format(p_a_0_2, p_a_1_2, p_a_2_2))
            print('Effect size: R2 = {:.5f}, R2.adj = {:.5f}'.format(R2_2, R2_adj_2))
            print('AIC = {:.2f}, AIC.c = {:.2f}'.format(AIC_2, AIC_c_2))
            print("-"*80)
           
        ###########################
        # Model 2a: y = a.1*x + a.2*x^2
        ###########################
        
        # First, add the x^2 column but do not add a constant a_0 to the model:
        X_2a = np.column_stack((x, x**2))
        model = sm.OLS(y,X_2a)
        results_2a = model.fit()
        a_0_2a = np.nan; a_1_2a = results_2a.params[0]; a_2_2a = results_2a.params[1]
        s_a_0_2a = np.nan; s_a_1_2a = results_2a.bse[0]; s_a_2_2a = results_2a.bse[1]
        p_a_0_2a = np.nan; p_a_1_2a = results_2a.pvalues[0]; p_a_2_2a = results_2a.pvalues[1]
        R2_2a = results_2a.rsquared; R2_adj_2a = results_2a.rsquared_adj
        AIC_2a, AIC_c_2a = DS_AIC(results_2a)
        s2_yx_2a = results_2a.scale
        V_2a = results_2a.cov_params()
        
        if(bVerbatim):
            print('Model ' + Models[Mod_d["2a"]] + ": " + Formulas[Mod_d["2a"]])
            print('Parameters:  a.0 = {:.3g}, a.1 = {:.3g}, a.2 = {:.3g}'.format(a_0_2a, a_1_2a, a_2_2a))
            print('P-values:    p(a.0) = {:.3g}, p(a.1) = {:.3g}, p(a.2) = {:.3g}'.format(p_a_0_2a, p_a_1_2a, p_a_2_2a))
            print('Effect size: R2 = {:.5f}, R2.adj = {:.5f}'.format(R2_2a, R2_adj_2a))
            print('AIC = {:.2f}, AIC.c = {:.2f}'.format(AIC_2a, AIC_c_2a))
            print("-"*80)
        
        ###########################
        # Model 2b: y = a.2*x^2
        ###########################
        
        # First, make the x^2 column but do not add a constant a_0 to the model:
        X_2b = x**2
        model = sm.OLS(y,X_2b)
        results_2b = model.fit()
        a_0_2b = np.nan; a_1_2b = np.nan; a_2_2b = results_2b.params[0]
        s_a_0_2b = np.nan; s_a_1_2b = np.nan; s_a_2_2b = results_2b.bse[0]
        p_a_0_2b = np.nan; p_a_1_2b = np.nan; p_a_2_2b = results_2b.pvalues[0]
        R2_2b = results_2b.rsquared; R2_adj_2b = results_2b.rsquared_adj
        AIC_2b, AIC_c_2b = DS_AIC(results_2b)
        s2_yx_2b = results_2b.scale
        V_2b = results_2b.cov_params()
        
        if(bVerbatim):
            print('Model ' + Models[Mod_d["2b"]] + ": " + Formulas[Mod_d["2b"]])
            print('Parameters:  a.0 = {:.3g}, a.1 = {:.3g}, a.2 = {:.3g}'.format(a_0_2b, a_1_2b, a_2_2b))
            print('P-values:    p(a.0) = {:.3g}, p(a.1) = {:.3g}, p(a.2) = {:.3g}'.format(p_a_0_2b, p_a_1_2b, p_a_2_2b))
            print('Effect size: R2 = {:.5f}, R2.adj = {:.5f}'.format(R2_2b, R2_adj_2b))
            print('AIC = {:.2f}, AIC.c = {:.2f}'.format(AIC_2b, AIC_c_2b))
            print("-"*80)
        
        ###########################
        # Model 2c: y = a.0 + a.2*x^2
        ###########################
        
        # First, make the x^2 column and a constant a_0 to the model:
        X_2c = x**2
        X_2c = sm.add_constant(X_2c)
        model = sm.OLS(y,X_2c)
        results_2c = model.fit()
        a_0_2c = results_2c.params[0]; a_1_2c = np.nan; a_2_2c = results_2c.params[1]
        s_a_0_2c = results_2c.bse[0]; s_a_1_2c = np.nan; s_a_2_2c = results_2c.bse[1]
        p_a_0_2c = results_2c.pvalues[0]; p_a_1_2c = np.nan; p_a_2_2c = results_2c.pvalues[1]
        R2_2c = results_2c.rsquared; R2_adj_2c = results_2c.rsquared_adj
        AIC_2c, AIC_c_2c = DS_AIC(results_2c)
        s2_yx_2c = results_2c.scale
        V_2c = results_2c.cov_params()
        
        if(bVerbatim):
            print('Model ' + Models[Mod_d["2c"]] + ": " + Formulas[Mod_d["2c"]])
            print('Parameters:  a.0 = {:.3g}, a.1 = {:.3g}, a.2 = {:.3g}'.format(a_0_2c, a_1_2c, a_2_2c))
            print('P-values:    p(a.0) = {:.3g}, p(a.1) = {:.3g}, p(a.2) = {:.3g}'.format(p_a_0_2c, p_a_1_2c, p_a_2_2c))
            print('Effect size: R2 = {:.5f}, R2.adj = {:.5f}'.format(R2_2c, R2_adj_2c))
            print('AIC = {:.2f}, AIC.c = {:.2f}'.format(AIC_2c, AIC_c_2c))
        
        
        ###########################
        # Collect relevant info in self:
        ###########################
        
        self.n = n
        
        self.results_0 = results_0; self.V_0 = V_0  # general results; var-covar par matrix
        self.results_1 = results_1; self.V_1 = V_1
        self.results_1a = results_1a; self.V_1a = V_1a
        self.results_2 = results_2; self.V_2 = V_2
        self.results_2a = results_2a; self.V_2a = V_2a
        self.results_2b = results_2b; self.V_2b = V_2b
        self.results_2c = results_2c; self.V_2c = V_2c
        
        self.Models = Models # names of models
        self.Mod_d = Mod_d # model dictionary
        self.Formulas = Formulas # formulas of models
        self.Ps = [1, 2, 1, 3, 2, 1, 2] # number of parameters of model
        self.AICs = [AIC_0, AIC_1, AIC_1a, AIC_2, AIC_2a, AIC_2b, AIC_2c]
        self.AIC_cs = [AIC_c_0,AIC_c_1, AIC_c_1a, AIC_c_2, AIC_c_2a, AIC_c_2b, AIC_c_2c]
        self.R2s = [R2_0, R2_1, R2_1a, R2_2, R2_2a, R2_2b, R2_2c]
        self.a_0s = [a_0_0, a_0_1, a_0_1a, a_0_2, a_0_2a, a_0_2b, a_0_2c]
        self.a_1s = [a_1_0, a_1_1, a_1_1a, a_1_2, a_1_2a, a_1_2b, a_1_2c]
        self.a_2s = [a_2_0, a_2_1, a_2_1a, a_2_2, a_2_2a, a_2_2b, a_2_2c]
        self.s_a_0s = [s_a_0_0, s_a_0_1, s_a_0_1a, s_a_0_2, s_a_0_2a, s_a_0_2b, s_a_0_2c]
        self.s_a_1s = [s_a_1_0, s_a_1_1, s_a_1_1a, s_a_1_2, s_a_1_2a, s_a_1_2b, s_a_1_2c]
        self.s_a_2s = [s_a_2_0, s_a_2_1, s_a_2_1a, s_a_2_2, s_a_2_2a, s_a_2_2b, s_a_2_2c]
        self.p_a_0s = [p_a_0_0, p_a_0_1, p_a_0_1a, p_a_0_2, p_a_0_2a, p_a_0_2b, p_a_0_2c]
        self.p_a_1s = [p_a_1_0, p_a_1_1, p_a_1_1a, p_a_1_2, p_a_1_2a, p_a_1_2b, p_a_1_2c]
        self.p_a_2s = [p_a_2_0, p_a_2_1, p_a_2_1a, p_a_2_2, p_a_2_2a, p_a_2_2b, p_a_2_2c]
        self.Xs = [X_0, X_1, X_1a, X_2, X_2a, X_2b, X_2c]
        self.s2_yxs = [s2_yx_0, s2_yx_1, s2_yx_1a, s2_yx_2, s2_yx_2a, s2_yx_2b, s2_yx_2c]
        
        pass;

    
    def summary(self):
        """
        *
        Method DS_CalibrationAnalysis.summary()
          
           Print a summary of all models in tabular form:
         
                Model_id, formula, AIC.c, Delta.c, w, R2, p-values parameters
        
        where  
                AIC.c        = small-sample Akaike Information Criterion
                Delta.c      = AIC.c (model) - lowest AIC.c-value
                w            = Akaike weight of model: w.m = exp(-Delta.m/2)/sum(exp(-Delta.j/2))_(j=1)^M
                R2           = R2-value (effect size of coefficient of determination)
        
           Author:      M.Emile F. Apol
           Date:        2022-09-14, revisions 2022-11-29
        """
        
        print('\n'+'='*80)
        print('SUMMARY of Calibration Models')
        print('='*80 + '\n')
        
        Models = self.Models
        Mod_d = self.Mod_d
        Formulas = self.Formulas
        AIC_cs = self.AIC_cs
        min_AIC_c = np.min(AIC_cs)
        Delta_cs = AIC_cs - min_AIC_c
        ws = np.exp(-0.5*Delta_cs)     # Akaike weights per model
        ws_tot = np.sum(ws)
        ws /= ws_tot
        R2s = self.R2s
        p_a_0s = self.p_a_0s
        p_a_1s = self.p_a_1s
        p_a_2s = self.p_a_2s
        
        # model formula AIC.c Delta.c w.c R2 p(a0) p(a1) p(a2)
        print_str1 = "{:<10} {:<30} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}"
        print_str2 = "{:<10} {:<30} {:>+10.2f} {:>10.2f} {:^10.3f} {:<10.5f} {:<10.3g} {:<10.3g} {:<10.3g}"
        
        print(print_str1.format('Model','Formula','AIC.c','Delta.c','w', 'R2','p(a.0)','p(a.1)','p(a.2)'))
        print(print_str1.format('----------','------------------------------','----------','----------','----------',
                                '----------','----------','----------','----------'))
        print(print_str2.format(Models[Mod_d["0"]],Formulas[Mod_d["0"]],AIC_cs[Mod_d["0"]],Delta_cs[Mod_d["0"]],ws[Mod_d["0"]],R2s[Mod_d["0"]],p_a_0s[Mod_d["0"]],p_a_1s[Mod_d["0"]],p_a_2s[Mod_d["0"]]))
        print(print_str2.format(Models[Mod_d["1"]],Formulas[Mod_d["1"]],AIC_cs[Mod_d["1"]],Delta_cs[Mod_d["1"]],ws[Mod_d["1"]],R2s[Mod_d["1"]],p_a_0s[Mod_d["1"]],p_a_1s[Mod_d["1"]],p_a_2s[Mod_d["1"]]))
        print(print_str2.format(Models[Mod_d["1a"]],Formulas[Mod_d["1a"]],AIC_cs[Mod_d["1a"]],Delta_cs[Mod_d["1a"]],ws[Mod_d["1a"]],R2s[Mod_d["1a"]],p_a_0s[Mod_d["1a"]],p_a_1s[Mod_d["1a"]],p_a_2s[Mod_d["1a"]]))
        print(print_str2.format(Models[Mod_d["2"]],Formulas[Mod_d["2"]],AIC_cs[Mod_d["2"]],Delta_cs[Mod_d["2"]],ws[Mod_d["2"]],R2s[Mod_d["2"]],p_a_0s[Mod_d["2"]],p_a_1s[Mod_d["2"]],p_a_2s[Mod_d["2"]]))
        print(print_str2.format(Models[Mod_d["2a"]],Formulas[Mod_d["2a"]],AIC_cs[Mod_d["2a"]],Delta_cs[Mod_d["2a"]],ws[Mod_d["2a"]],R2s[Mod_d["2a"]],p_a_0s[Mod_d["2a"]],p_a_1s[Mod_d["2a"]],p_a_2s[Mod_d["2a"]]))
        print(print_str2.format(Models[Mod_d["2b"]],Formulas[Mod_d["2b"]],AIC_cs[Mod_d["2b"]],Delta_cs[Mod_d["2b"]],ws[Mod_d["2b"]],R2s[Mod_d["2b"]],p_a_0s[Mod_d["2b"]],p_a_1s[Mod_d["2b"]],p_a_2s[Mod_d["2b"]]))
        print(print_str2.format(Models[Mod_d["2c"]],Formulas[Mod_d["2c"]],AIC_cs[Mod_d["2c"]],Delta_cs[Mod_d["2c"]],ws[Mod_d["2c"]],R2s[Mod_d["2c"]],p_a_0s[Mod_d["2c"]],p_a_1s[Mod_d["2c"]],p_a_2s[Mod_d["2c"]]))
        pass;
    
    
   
    def results_model(self, model):
        """
        *
        Method DS_CalibrationAnalysis.results_model(model)
          
           Select the regression results of the requested calibration model model_id:
         
           Input:
              model                   preferred model: string ["0", "1", "1a", "2", "2a", "2b", "2c"]
              
           Returns:
              OLS object with results of the regression
          
           Author:      M. Emile F. Apol
           Date:        2022-09-15
        """
        
        if model == "0":
            res = self.results_0
        elif model == "1":
            res = self.results_1
        elif model == "1a":
            res = self.results_1a
        elif model == "2":
            res = self.results_2
        elif model == "2a":
            res = self.results_2a
        elif model == "2b":
            res = self.results_2b
        elif model == "2c":
            res = self.results_2c
        else:
            print("Wrong model chosen!")
            res = None
            return()
        
        return(res)
    
    
    def predict(self, xp, model, confidence=0.95, bVerbatim=False, bDebug=False):
        """
        *
        Method DS_CalibrationAnalysis.predict(xp, model, confidence=0.95, bVerbatim=False, bDebug=False)
        
          Predict y.p of the requested calibration model with "confidence"*100% CI 
          for given x.p value
         
          General formula:
         
             s_yp^2 = xp^T * V * xp
         
             CI(yp) = t(n-P) * s_yp
         
          Reference: Montgomery & Runger (2011), Eq. (12-39)
          
          Input:
             xp                      value of x for prediction
             model                   preferred model: string ["0", "1", "1a", "2", "2a", "2b", "2c"]
             confidence              [default: 0.95]
             bVerbatim               True: print results to stdout
         
          Returns:                   yp, yp_lower, yp_upper
          
          Author:     M. Emile F. Apol
          Date:       2022-09-15
        """
        
        # Get the shared libraries:
        t = self.t
        np = self.np
        
        Err = 0
        
        # Get the required shared data:
        n = self.n
        Ps = self.Ps
        Models = self.Models
        Mod_d = self.Mod_d
        a_0s = self.a_0s
        a_1s = self.a_1s
        a_2s = self.a_2s
        
        if((confidence >= 0) & (confidence <=1)):
            alpha = 1 - confidence
            conf_str = "{:g}".format(100*confidence)
        else:
            print('Wrong confidence chosen! Must be between 0 and 1...')
            Err = 1;
            return()
        
        if(Err != 1):
            
            if(model=="0"):
                a = np.array([a_0s[Mod_d["0"]]])
                V = self.V_0
                Xp = np.array([1])
                yp = np.dot(a, Xp)
                s_yp = np.sqrt(np.dot(np.matmul(Xp, V), Xp))
                df = n-Ps[Mod_d["0"]]
                t_val = t.ppf(1-alpha/2, df)
                yp_lower = yp - t_val*s_yp
                yp_upper = yp + t_val*s_yp
                if(bDebug):
                    print("a: ", a)
                    print("x: ", Xp)
                    print("V: ", V)
                    print("yp: {:.4f}, CI = [{:.4f}, {:.4f}]".format(yp, yp_lower, yp_upper))
        
            elif(model=="1"):
                a = np.array([a_0s[Mod_d["1"]], a_1s[Mod_d["1"]]])
                V = self.V_1
                Xp = np.array([1, xp])
                yp = np.dot(a, Xp)
                s_yp = np.sqrt(np.dot(np.matmul(Xp, V), Xp))
                df = n-Ps[Mod_d["1"]]
                t_val = t.ppf(1-alpha/2, df)
                yp_lower = yp - t_val*s_yp
                yp_upper = yp + t_val*s_yp
                if(bDebug):
                    print("a: ", a)
                    print("x: ", Xp)
                    print("V: ", V)
                    print("yp: {:.4f}, CI = [{:.4f}, {:.4f}]".format(yp, yp_lower, yp_upper))
            
            elif(model=="1a"):
                a = np.array([a_1s[Mod_d["1"]]])
                V = self.V_1a
                Xp = np.array([xp])
                yp = np.dot(a, Xp)
                s_yp = np.sqrt(np.dot(np.matmul(Xp, V), Xp))
                df = n-Ps[Mod_d["1a"]]
                t_val = t.ppf(1-alpha/2, df)
                yp_lower = yp - t_val*s_yp
                yp_upper = yp + t_val*s_yp
                if(bDebug):
                    print("a: ", a)
                    print("x: ", Xp)
                    print("V: ", V)
                    print("yp: {:.4f}, CI = [{:.4f}, {:.4f}]".format(yp, yp_lower, yp_upper))
            
            elif(model=="2"):
                a = np.array([a_0s[Mod_d["2"]], a_1s[Mod_d["2"]], a_2s[Mod_d["2"]]])
                V = self.V_2
                Xp = np.array([1, xp, xp**2])
                yp = np.dot(a, Xp)
                s_yp = np.sqrt(np.dot(np.matmul(Xp, V), Xp))
                df = n-Ps[Mod_d["2"]]
                t_val = t.ppf(1-alpha/2, df)
                yp_lower = yp - t_val*s_yp
                yp_upper = yp + t_val*s_yp
                if(bDebug):
                    print("a: ", a)
                    print("x: ", Xp)
                    print("V: ", V)
                    print("yp: {:.4f}, CI = [{:.4f}, {:.4f}]".format(yp, yp_lower, yp_upper))
            
            elif(model=="2a"):
                a = np.array([a_1s[Mod_d["2a"]], a_2s[Mod_d["2a"]]])
                V = self.V_2a
                Xp = np.array([xp, xp**2])
                yp = np.dot(a, Xp)
                s_yp = np.sqrt(np.dot(np.matmul(Xp, V), Xp))
                df = n-Ps[Mod_d["2a"]]
                t_val = t.ppf(1-alpha/2, df)
                yp_lower = yp - t_val*s_yp
                yp_upper = yp + t_val*s_yp
                if(bDebug):
                    print("a: ", a)
                    print("x: ", Xp)
                    print("V: ", V)
                    print("yp: {:.4f}, CI = [{:.4f}, {:.4f}]".format(yp, yp_lower, yp_upper))
            
            elif(model=="2b"):
                a = np.array([a_2s[Mod_d["2b"]]])
                V = self.V_2b
                Xp = np.array([xp**2])
                yp = np.dot(a, Xp)
                s_yp = np.sqrt(np.dot(np.matmul(Xp, V), Xp))
                df = n-Ps[Mod_d["2b"]]
                t_val = t.ppf(1-alpha/2, df)
                yp_lower = yp - t_val*s_yp
                yp_upper = yp + t_val*s_yp
                if(bDebug):
                    print("a: ", a)
                    print("x: ", Xp)
                    print("V: ", V)
                    print("yp: {:.4f}, CI = [{:.4f}, {:.4f}]".format(yp, yp_lower, yp_upper))
            
            elif(model=="2c"):
                a = np.array([a_0s[Mod_d["2c"]], a_2s[Mod_d["2c"]]])
                V = self.V_2c
                Xp = np.array([1, xp**2])
                yp = np.dot(a, Xp)
                s_yp = np.sqrt(np.dot(np.matmul(Xp, V), Xp))
                df = n-Ps[Mod_d["2c"]]
                t_val = t.ppf(1-alpha/2, df)
                yp_lower = yp - t_val*s_yp
                yp_upper = yp + t_val*s_yp
                if(bDebug):
                    print("a: ", a)
                    print("x: ", Xp)
                    print("V: ", V)
                    print("yp: {:.4f}, CI = [{:.4f}, {:.4f}]".format(yp, yp_lower, yp_upper))
            
            else:
                print("Wrong model chosen!")
                Err = 1
                return()
            
        if(Err==1):
            return()
        else:
            if(bVerbatim):
                print("Predicted value at x = {:.4g}:".format(xp))
                print("yp = {:.4g}, ".format(yp) + conf_str + "% CI = [{:.4g}, {:.4g}]".format(yp_lower, yp_upper))
                
            return(yp, yp_lower, yp_upper);
    
    
            
    def plot_model(self, model, confidence=0.95, **kwargs):
        """
        *
        Method DS_CalibrationAnalysis.plot_model(self, model, confidence=0.95, **kwargs)
          
           Make a plot of the experimental data with the requested calibration model and CI.
         
           Input:
              model                   preferred model: string ["0", "1", "1a", "2", "2a", "2b", "2c"]
              confidence              confidence level [default: 0.95]
          
           Optional arguments:
              xlabel, ylabel          labels along the x- and y-axis [def: "Independent factor, x", "Response, y"]
              title                   Main title above the plot [def: Calibration curve]
              xlim, ylim              lists [x_min, x_max] and/or [y_min, y_max] minimum 
                                              and maximum values along the axes
              data_color, data_marker color and type of marker for experimental points [def: 'black', 'o']
              line_color, line_style  line color and style for calibration model [def: "red", '-']
              CI_color, CI_style      line color and style for CI lines [def: 'blue', '--']
          
           Author:      M.Emile F. Apol
           Date:        2022-09-22
        """
        
        # Get the libraries:
        plt = self.plt
        t = self.t
        
        Err = 0
        
        if((confidence >= 0) & (confidence <=1)):
            alpha = 1 - confidence
        else:
            print('Wrong confidence chosen! Must be between 0 and 1...')
            Err = 1;
        
        # First, get possible extra plotting arguments:
        # xlabel, ylabel, title, xlim, ylim, data_color, data_marker, line_color
        xlabel = kwargs.get('xlabel', 'Independent factor, $x$')
        ylabel = kwargs.get('ylabel', 'Response, $y$')
        title = kwargs.get('title', 'Calibration curve')
        xlim = kwargs.get('xlim', None)
        ylim = kwargs.get('ylim', None)
        data_color = kwargs.get('data_color', 'black')
        data_marker = kwargs.get('data_marker', 'o')
        line_color = kwargs.get('line_color', 'red')
        line_style = kwargs.get('line_style', '-')
        CI_color = kwargs.get('CI_color', 'blue')
        CI_style = kwargs.get('CI_style', '--')
        
        # get the relevant info:
        x = self.x
        y = self.y
        n = self.n
        Ps = self.Ps
        Models = self.Models
        Mod_d = self.Mod_d
        
        if xlim is None:
            x_min = np.min(x); x_max = np.max(x)
        else:
            x_min = xlim[0]
            x_max = xlim[1]

        # Plot the best model in a graph:
        
        n_plot = 101
        xn = np.linspace(x_min, x_max, n_plot)
        y_pred  = []
        y_lower = []
        y_upper = []
        
        # Calculate predictions per model:
        
        for xi in xn:
            y_temp, lower_temp, upper_temp = self.predict(xi, model=model, confidence=1-alpha, bVerbatim=False)
            y_pred.append(y_temp)
            y_lower.append(lower_temp)
            y_upper.append(upper_temp)
            
        # CHECK FOR WRONG MODEL!
        
        # Generic plotting of data and model:
        if(Err != 1):
            conf_str = "{:g}".format(100*confidence)
            plt.scatter(x, y, marker=data_marker, color=data_color, label='Experimental')
            plt.plot(xn, y_pred, color=line_color, linestyle=line_style, 
                     label='Model ' + Models[Mod_d[model]])
            plt.plot(xn, y_lower, color=CI_color, linestyle=CI_style, label=conf_str + "% CI")
            plt.plot(xn, y_upper, color=CI_color, linestyle=CI_style)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend(loc='best')
            plt.title(title)
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.show()

        pass;
    
         
    def interpolate(self, y_0, model, confidence=0.95, bVerbatim=True, bDebug = False, **kwargs):
        """
        *
        Method DS_CalibrationAnalysis.interpolate(y_0, model, confidence=0.95, bVerbatim=True, bDebug = False, **kwargs)
          
           Interpolate measured y.0-values (m replicas) with the requested calibration model 
           towards an x.0-value, with standard error and CI.
         
           General formula:
         
               s_x0^2 = (s_yx^2 / m) * (dx0/dy0)^2 + (dx0/da)^T * V * (dx0/da)
         
               CI(x0) = t(n-P) * s_x0
          
           Input:
               y_0                     array of (replica) measurements of y-values of a single sample
               model                   preferred model: string ["0", "1", "1a", "2", "2a", "2b", "2c"]
               confidence              confidence level [default: 0.95]
               bVerbatim               True: print results also to stdout
                                       False: no output to stdout
          
           Optional arguments:
         
           Return:                     x_0_star, x_0_lower, x_0_upper, s_x_0, df, y_0_av, m
          
           Author:          M. Emile F. Apol
           Date:            2022-09-25

        """
        
        # Get the libraries:
        plt = self.plt
        t = self.t
        np = self.np
        
        # Get the relevant info:
        x = self.x
        y = self.y
        n = self.n
        Ps = self.Ps
        Models = self.Models
        Mod_d = self.Mod_d  
        a_0s = self.a_0s
        a_1s = self.a_1s
        a_2s = self.a_2s
        s2_yxs = self.s2_yxs
        Xs = self.Xs
        
        # Check if y_0 is an array, if a scaler make it an array:
        if(not isinstance(y_0, (np.ndarray, list))):
            y_0 = np.array([y_0])
        else:
            y_0 = np.array(y_0)
        
        Err = 0
        
        if((confidence >= 0) & (confidence <=1)):
            alpha = 1 - confidence
        else:
            print('Wrong confidence chosen! Must be between 0 and 1...')
            Err = 1;
            return()
        
        if(Err != 1):
            
            # Determine the middle of the calibration x-data:
            xc_min = np.min(x)
            xc_max = np.max(x)
            xc_mid = (xc_max - xc_min)/2
            
            # Determine properties of replica y0-values:
            y_0 = np.array(y_0)     # to be sure
            m = len(y_0)            # nr of replicas
            y_0_av = np.mean(y_0)   # mean measured respons of sample
            
            # Do interpolation per model:
        
            if(model=="1"):
                a_0 = a_0s[Mod_d["1"]]; a_1 = a_1s[Mod_d["1"]]
                s2_yx = s2_yxs[Mod_d["1"]]
                x_0 = (y_0_av - a_0) / a_1
                x_0_star = x_0
                V = self.V_1
                dx0dy0 = (1 / a_1)
                dx0da = (-1 / a_1) * np.array([1, x_0_star])
                s_x_0 = np.sqrt((s2_yx / m) * dx0dy0**2 + np.dot(np.matmul(dx0da, V), dx0da))
                df = n-Ps[Mod_d["1"]]
                t_val = t.ppf(1-alpha/2, df)
                x_0_lower = x_0_star - t_val*s_x_0
                x_0_upper = x_0_star + t_val*s_x_0
                
            
            elif(model=="1a"):
                a_1 = a_1s[Mod_d["1a"]]
                s2_yx = s2_yxs[Mod_d["1a"]]
                x_0 = (y_0_av) / a_1
                x_0_star = x_0
                V = self.V_1a
                dx0dy0 = (1 / a_1)
                dx0da = (-1 / a_1) * np.array([x_0_star])
                s_x_0 = np.sqrt((s2_yx / m) * dx0dy0**2 + np.dot(np.matmul(dx0da, V), dx0da))
                df = n-Ps[Mod_d["1a"]]
                t_val = t.ppf(1-alpha/2, df)
                x_0_lower = x_0_star - t_val*s_x_0
                x_0_upper = x_0_star + t_val*s_x_0
            
            elif(model=="2"):
                a_0 = a_0s[Mod_d["2"]]; a_1 = a_1s[Mod_d["2"]]; a_2 = a_2s[Mod_d["2"]]
                s2_yx = s2_yxs[Mod_d["2"]]
                x_0_1 = (-a_1 + np.sqrt(a_1**2 + 4*a_2*(y_0_av-a_0)))/(2*a_2)
                x_0_2 = (-a_1 - np.sqrt(a_1**2 + 4*a_2*(y_0_av-a_0)))/(2*a_2)
                if(np.abs(x_0_1-xc_mid) < np.abs(x_0_2-xc_mid)):
                    x_0_star = x_0_1
                else:
                    x_0_star = x_0_2
                V = self.V_2
                dx0dy0 = (1 / (a_1+2*a_2*x_0_star))
                dx0da = (-1 / (a_1+2*a_2*x_0_star)) * np.array([1, x_0_star, x_0_star**2])
                s_x_0 = np.sqrt((s2_yx / m) * dx0dy0**2 + np.dot(np.matmul(dx0da, V), dx0da))
                df = n-Ps[Mod_d["2"]]
                t_val = t.ppf(1-alpha/2, df)
                x_0_lower = x_0_star - t_val*s_x_0
                x_0_upper = x_0_star + t_val*s_x_0
            
            elif(model=="2a"):
                a_1 = a_1s[Mod_d["2a"]]; a_2 = a_2s[Mod_d["2a"]]
                s2_yx = s2_yxs[Mod_d["2a"]]
                x_0_1 = (-a_1 + np.sqrt(a_1**2 + 4*a_2*(y_0_av)))/(2*a_2)
                x_0_2 = (-a_1 - np.sqrt(a_1**2 + 4*a_2*(y_0_av)))/(2*a_2)
                if(np.abs(x_0_1-xc_mid) < np.abs(x_0_2-xc_mid)):
                    x_0_star = x_0_1
                else:
                    x_0_star = x_0_2
                V = self.V_2a
                dx0dy0 = (1 / (a_1+2*a_2*x_0_star))
                dx0da = (-1 / (a_1+2*a_2*x_0_star)) * np.array([x_0_star, x_0_star**2])
                s_x_0 = np.sqrt((s2_yx / m) * dx0dy0**2 + np.dot(np.matmul(dx0da, V), dx0da))
                df = n-Ps[Mod_d["2a"]]
                t_val = t.ppf(1-alpha/2, df)
                x_0_lower = x_0_star - t_val*s_x_0
                x_0_upper = x_0_star + t_val*s_x_0
            
            elif(model=="2b"):
                a_2 = a_2s[Mod_d["2b"]]
                s2_yx = s2_yxs[Mod_d["2b"]]
                x_0_1 = np.sqrt(y_0_av / a_2)
                x_0_2 = -1*np.sqrt(y_0_av / a_2)
                if(np.abs(x_0_1-xc_mid) < np.abs(x_0_2-xc_mid)):
                    x_0_star = x_0_1
                else:
                    x_0_star = x_0_2
                V = self.V_2b
                dx0dy0 = (1 / (2*a_2*x_0_star))
                dx0da = (-1 / (2*a_2*x_0_star)) * np.array([x_0_star**2])
                s_x_0 = np.sqrt((s2_yx / m) * dx0dy0**2 + np.dot(np.matmul(dx0da, V), dx0da))
                df = n-Ps[Mod_d["2b"]]
                t_val = t.ppf(1-alpha/2, df)
                x_0_lower = x_0_star - t_val*s_x_0
                x_0_upper = x_0_star + t_val*s_x_0
            
            elif(model=="2c"):
                a_0 = a_0s[Mod_d["2c"]]; a_2 = a_2s[Mod_d["2c"]]
                s2_yx = s2_yxs[Mod_d["2c"]]
                x_0_1 = np.sqrt((y_0_av-a_0) / a_2)
                x_0_2 = -1*np.sqrt((y_0_av-a_0) / a_2)
                if(np.abs(x_0_1-xc_mid) < np.abs(x_0_2-xc_mid)):
                    x_0_star = x_0_1
                else:
                    x_0_star = x_0_2
                V = self.V_2c
                dx0dy0 = (1 / (2*a_2*x_0_star))
                dx0da = (-1 / (2*a_2*x_0_star)) * np.array([1, x_0_star**2])
                s_x_0 = np.sqrt((s2_yx / m) * dx0dy0**2 + np.dot(np.matmul(dx0da, V), dx0da))
                df = n-Ps[Mod_d["2c"]]
                t_val = t.ppf(1-alpha/2, df)
                x_0_lower = x_0_star - t_val*s_x_0
                x_0_upper = x_0_star + t_val*s_x_0
            
            else:
                print("Wrong model chosen!")
                Err = 1
                return()
            
            if(bDebug and (Err !=1)):
                    print("y0: ", y_0_av, ", m: ",m)
                    print("s2_yx: ", s2_yx)
                    print("dx0dy0: ", dx0dy0)
                    print("dx0da:  ", dx0da)
                    print("V: ", V)
                    print("V: ", V_alt)
                    print("x0: ", x_0_star)
                    print("s2_x0 (1): ", (s2_yx / m) * dx0dy0**2)
                    print("s2_x0 (2): ", np.dot(np.matmul(dx0da, V), dx0da))
                    print("df: ", df)
                    print("x0: {:.4f}, 95% CI = [{:.4f}, {:.4f}]".format(x_0_star, x_0_lower, x_0_upper))
            
            # print results to stdout:
            if(bVerbatim):
                conf_str = "{:g}".format(100*confidence)
                print("Interpolation of model " + model + ":")
                print('y.0 = {:.4g}, m = {:d} replica(s)'.format(y_0_av, m))
                print('x.0 = {:.4g}, '.format(x_0_star) + 
                     conf_str + "% CI = [{:.4g}, {:.4g}]".format(x_0_lower, x_0_upper))
                print('s.x.0 = {:.4g}'.format(s_x_0))
            
            
        if(Err==1):
            return()
        else:
            return(x_0_star, x_0_lower, x_0_upper, s_x_0, df, y_0_av, m);
    


# ## 23. Calculate the ${\rm AIC}$ and ${\rm AIC_c}$-values from a statsmodels.api OLS fit

# In[108]:


def DS_OLS_AIC(results):
    """
    *
    Function DS_OLS_AIC(results)
    
      This function calculates the Akaike AIC and small-sample Akaike AIC.c-values 
      of a LS linear regression model based on the object from a statsmodels.api.OLS.fit().
     
      Equations:
     
      AIC = n*log(SS.err/n) + 2*(P+1)
      AIC.c = n*log(SS.err/n) + 2*(P+1) + 2*(P+1)*(P+2)/(n-P-2)
      
      where n = number of observations, P = number of model parameters, and 
         SS.err = residual sum of squares
         
    Requires:      numpy as np 
    
    Input:         results       results of a statsmodels.api.OLS.fit()
    
    Return:        AIC, AIC.c    Akaike value, small-sample Akaike value
                   N.B. If n-P-2 < 1, AIC.c will return np.nan
    
    Author:        M.E.F. Apol
    Date:          2020-01-06, update: 2022-10-31
    """
    
    SS_err = results.ssr
    n = results.nobs
    P = len(results.params)
    AIC = n * np.log(SS_err/n) + 2 * (P + 1)
    if(n-P-2 >0):
        AIC_c = AIC + 2 * (P + 1) * (P + 2) / (n - P - 2)
    else:
        AIC_c = np.nan
    return(AIC, AIC_c);


# ## 24. Calculate the predicted $y$-values with CI from a statsmodels.api OLS fit

# In[111]:


def DS_OLS_predict_with_CI(results, X, confidence=0.95):
    """
    *
    Function DS_OLS_predict_with_CI(results, X, confidence=0.95)
    
      This function calculates the model predictions y_pred with 
      (100*confidence)% Confidence Interval (lower and upper limits) 
      of a LS linear regression model based on the object from a statsmodels.api.OLS.fit().
    
    Requires:           numpy as np
    
    Input: 
        results         results of a statsmodels.api.OLS.fit()
        X               design matrix for model using predicting x-values
        confidence=0.95 confidence level of CI [default 95%]
    
    Return:
        y_pred          predicted y-value
        y_pred_lower    lower confidence*100% CI
        y_pred_upper    upper confidence*100% CI
     
    Author: M.E.F. Apol
    Date:   30-10-2022
    """
    
    from scipy.stats import t
    
    a_hat = results.params
    P = len(a_hat)
    V = results.cov_params()
    n = results.nobs
    
    y_pred = []
    y_pred_lower = []
    y_pred_upper = []
    t_vals = t.ppf([0.025, 0.975], n-P)

    # Be sure that X is a 2d array:
    if (np.ndim(X) == 1):
        X = np.reshape(X, (len(X), 1))
    n_plot = len(X[:,0])
        
    # Calculate the prediction of the model with 100*confidence% CI:
    for i in range(n_plot):
        x_p = X[i,:]
        y_p = np.dot(x_p, a_hat)
        s_y_p = np.sqrt(np.linalg.multi_dot((x_p, V, x_p)))
        y_p_lower, y_p_upper = y_p + t_vals * s_y_p
        y_pred.append(y_p)
        y_pred_lower.append(y_p_lower)
        y_pred_upper.append(y_p_upper)
        
    return(y_pred, y_pred_lower, y_pred_upper)


# ## <span style="color:blue">*Other extra functions:*</span>

# ## 25. Make a 1d or 2d contingency table from 1 or 2 arrays

# In[115]:


def DS_xtab(*cols, apply_wt=False):
    
    ################################################################################
    #
    # Define a function to generate a (1-way or) 2-way contingency table from
    # (a single or) two arrays of equal length.
    #   assuming a normal approximation
    #
    # Author:            Doug Ybarbo, adaptation M.E.F. Apol
    # Source:            https://gist.github.com/alexland/d6d64d3f634895b9dc8e
    # Date:              2021-11-30
    # Usage:             my_xtab(y1 [, y2]) 
    #
    #                    Note: y is an array with BINARY data (0, 1)
    #
    # Return:            uv, table
    #                    where
    #                       uv = tuple containing the row and column headers
    #                       table = contingency table
    #
    ################################################################################
    '''
  Source: https://gist.github.com/alexland/d6d64d3f634895b9dc8e  (Doug Ybarbo)
  Adapted by M.E.F. Apol, 2021-11-30
  returns:
    (i) xt, numpy array storing the xtab results, number of dimensions is equal to 
        the len(args) passed in
    (ii) unique_vals_all_cols, a tuple of 1D numpy array for each dimension 
        in xt (for a 2D xtab, the tuple comprises the row and column headers)
    pass in:
      (i) 1 or more 1D numpy arrays of integers
      (ii) if wts is True, then the last array in cols is an array of weights
      
  if return_inverse=True, then np.unique also returns an integer index 
  (from 0, & of same len as array passed in) such that, uniq_vals[idx] gives the original array passed in
  higher dimensional cross tabulations are supported (eg, 2D & 3D)
  cross tabulation on two variables (columns):
  >>> q1 = NP.array([7, 8, 8, 8, 5, 6, 4, 6, 6, 8, 4, 6, 6, 6, 6, 8, 8, 5, 8, 6])
  >>> q2 = NP.array([6, 4, 6, 4, 8, 8, 4, 8, 7, 4, 4, 8, 8, 7, 5, 4, 8, 4, 4, 4])
  >>> uv, xt = xtab(q1, q2)
  >>> uv
    (array([4, 5, 6, 7, 8]), array([4, 5, 6, 7, 8]))
  >>> xt
    array([[2, 0, 0, 0, 0],
           [1, 0, 0, 0, 1],
           [1, 1, 0, 2, 4],
           [0, 0, 1, 0, 0],
           [5, 0, 1, 0, 1]], dtype=uint64)
    '''

    import numpy as np
    
    if not all(len(col) == len(cols[0]) for col in cols[1:]):
        raise ValueError("all arguments must be same size")

    if len(cols) == 0:
        raise TypeError("xtab() requires at least one argument")

    fnx1 = lambda q: len(q.squeeze().shape)
    if not all([fnx1(col) == 1 for col in cols]):
        raise ValueError("all input arrays must be 1D")

    if apply_wt:
        cols, wt = cols[:-1], cols[-1]
    else:
        wt = 1

    uniq_vals_all_cols, idx = zip( *(np.unique(col, return_inverse=True) for col in cols) )
    shape_xt = [uniq_vals_col.size for uniq_vals_col in uniq_vals_all_cols]
    dtype_xt = 'float' if apply_wt else 'uint'
    xt = np.zeros(shape_xt, dtype=dtype_xt)
    np.add.at(xt, idx, wt)
    return uniq_vals_all_cols, xt


# ## 26. Beta-Negative Binomial distribution

# The Beta-Negative Binomial distribution has pmf:
# 
# \begin{equation}
# f(y | r, a, b) = \frac{\Gamma(r+y)}{\Gamma(r) \cdot y!} \cdot \frac{B(a+r, b+y)}{B(a, b)}
# \end{equation}
# 
# where
# 
# \begin{equation}
# y! = \Gamma(y+1)
# \end{equation}
# 
# so, for computational stability, use
# 
# \begin{equation}
# \ln (f(y)) = \ln \Gamma(r+y) - \ln \Gamma(r) - \ln \Gamma(y+1) + \ln B(a+r, b+y) - \ln B(a,b)
# \end{equation}
# 
# where $\ln \Gamma(a)$ is `gammaln(a)` and $\ln B(a,b)$ is `betaln(a,b)` from `scipy.special`, and calculate $f(y) = e^{ \ln (f(y) )}$.
# 
# For the cdf $F(x)$ simply sum all the pmf-values from $y=0$ to $y=x$.
# 
# Random variables can be generated from random Beta and random Negative Binomial variables:
# 
# \begin{equation}
# p \sim Beta(a, b) \\
# y \sim NBin(r, p)
# \end{equation}
# 
# where $p$ is generated using `beta.rvs(a, b, size)` and $y$ is generated using `nbinom.rvs(r, p, size)` from `scipy.stats`.
# 
# For the quantile (ppf) function, use the `scipy.stats` definition for discrete distributions, that the quantile $Q_p$ is the smallest value $Q_p$ such that $F(Q_p) \geq p$. Default value for $p=0$ is $Q_p = -1$.

# In[117]:


from scipy.special import gammaln 
from scipy.special import betaln 
from scipy.stats import beta
from scipy.stats import nbinom
import numpy as np

class DS_Beta_NBinom:
    """
    *
    Function DS_Beta_NBinom
    
       This function calculates properties of the Beta-Negative Binomial distribution
       Beta-NBin(r, a, b), in absense of a similar function in scipy.stats...
       
       Methods:
       --------
       DS_Beta_NBinom(r, a, b).pmf(y)       probability mass function f(y)
       DS_Beta_NBinom(r, a, b).cdf(x)       cumulative distribution function F(x)
       DS_Beta_NBinom(r, a, b).ppf(p)       quantile function Q(p) = F^(-1)(p)
       DS_Beta_NBinom(r, a, b).rvs(size)    n = size random variables
       
       Author: M. Emile F. Apol
       Date:   2022-11-22
    """
    
    def __init__(self, r, a, b):
        
        self.r = r
        self.a = a
        self.b = b
        
        pass;
    
        
    def pmf(self, y):
        """
        * 
        Function DS_Beta_NBinom(r, a, b).pmf(y)
        
           This function calculates the probability mass function f(y) of the 
           Beta-Negative Binomial Beta-NBin(r, a, b) distribution. 
           
           Return: f(y) value
           
           Author: M. Emile F. Apol
           Date:   2022-08-31, rev. 2022-11-22
        """
        r = self.r; a = self.a; b = self.b
        
        lnf = gammaln(r+y) - gammaln(r) - gammaln(y+1) + betaln(a+r, b+y) - betaln(a, b)
        f = np.exp(lnf)
        return(f)
    
    
    def cdf(self, x):
        """
        * 
        Function DS_Beta_NBinom(r, a, b).cdf(x)
        
           This function calculates the cumulative probability distribution F(x) of the 
           Beta-Negative Binomial Beta-NBin(r, a, b) distribution.
           
           Return: F(x) value
           
           Author: M. Emile F. Apol
           Date:   2022-08-31, rev. 2022-11-22
        """
        r = self.r; a = self.a; b = self.b
        
        x_help = np.arange(0, x+1)
        f_help = np.array([self.pmf(xi) for xi in x_help])
        cdf = np.sum(f_help)
        return(cdf)
    
    
    def rvs(self, size=1):
        """
        * 
        Function DS_Beta_NBinom(r, a, b).rvs(size)
        
           This function calculates n = size random variables from a 
           Beta-Negative Binomial Beta-NBin(r, a, b) distribution. 
           
           Return: 1d array of values
           
           Author: M. Emile F. Apol
           Date:   2022-08-31, rev. 2022-11-22
        """
        r = self.r; a = self.a; b = self.b
        
        p = beta.rvs(a, b, loc=0, scale=1, size=size)
        x = np.array([nbinom.rvs(r, pii, loc=0, size=1) for pii in p])
        # Flatten the resulting array to a list
        x = np.concatenate(x).ravel().tolist()
        return(x)
    

    def ppf(self, p):
        """
        * 
        Function DS_Beta_NBinom(r, a, b).ppf(p)
        
           This function calculates the quantile Q.p = F^(-1)(p) from a 
           Beta-Negative Binomial Beta-NBin(r, a, b) distribution. 
           
           Return: quantile Q.p
           
           Author: M. Emile F. Apol
           Date:   2022-08-31, rev. 2022-11-22
        """
        r = self.r; a = self.a; b = self.b
        
        # First check whether p is a list or a single value
        bFloat = bInt = False
        if type(p) is float:
            p = [p]
            bFloat = True
        elif type(p) is int:
            p = [p]
            bInt = True
        elif type(p) is list:
            p = p
        else:
            return()
        # Do a loop over all values of p
        quantile = list()
        for pii in p:
            # Convention in scipy.stats: if p = 0, x = -1
            x = -1
            if pii==0.0:
                res = x
                quantile.append(res)
            else:
                bDoStep = True
                while bDoStep:
                    x = x + 1
                    F = self.cdf(x)
                    if F < pii:
                        bDoStep = True
                    else:
                        bDoStep = False
                res = x
                quantile.append(res)
        if bFloat or bInt:
            quantile = quantile[0]
        return(quantile)


# ## 27. Calculate the central sample moment $m_r$ from a 1d array 

# In[125]:


def DS_Mu_r(y, r):
    """
    *
    Function DS_Mu_r(y, r)
     
       This function calculates m.r, the r.th central moment of a 1d numpy array y.
    
    Requires:          numpy as np
     
    Input:
      y                data vector
      r                power
     
    Return:            mu.r[y] : the rth central moment of y
    
    Author:            M.E.F. Apol
    Date:              2020-01-06
    """
    
    y_av = np.mean(y)
    mu_r = np.mean((y - y_av)**r)
    return(mu_r);

