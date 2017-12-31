# decomposethepysal
拆轮子
## 这个项目是为了学习使用github与写出更为pythonic的代码而建立的。
##这里我先读一下我使用最多的bivariate local moran I


class Moran_Local_BV(object):
    """Bivariate Local Moran Statistics


    Parameters
    ----------
    x : array
        x-axis variable
    y : array
        (n,1), wy will be on y axis
    w : W
        weight instance assumed to be aligned with y
    transformation : {'R', 'B', 'D', 'U', 'V'}
                     weights transformation,  default is row-standardized "r".
                     Other options include
                     "B": binary,
                     "D": doubly-standardized,
                     "U": untransformed (general weights),
                     "V": variance-stabilizing.
    permutations   : int
                     number of random permutations for calculation of pseudo
                     p_values
    geoda_quads    : boolean
                     (default=False)
                     If True use GeoDa scheme: HH=1, LL=2, LH=3, HL=4
                     If False use PySAL Scheme: HH=1, LH=2, LL=3, HL=4

    Attributes
    ----------

    zx           : array
                   original x variable standardized by mean and std
    zy           : array
                   original y variable standardized by mean and std
    w            : W
                   original w object
    permutations : int
                   number of random permutations for calculation of pseudo
                   p_values
    Is           : float
                   value of Moran's I
    q            : array
                   (if permutations>0)
                   values indicate quandrant location 1 HH,  2 LH,  3 LL,  4 HL
    sim          : array
                   (if permutations>0)
                   vector of I values for permuted samples
    p_sim        : array
                   (if permutations>0)
                   p-value based on permutations (one-sided)
                   null: spatial randomness
                   alternative: the observed Ii is further away or extreme
                   from the median of simulated values. It is either extremelyi
                   high or extremely low in the distribution of simulated Is.
    EI_sim       : array
                   (if permutations>0)
                   average values of local Is from permutations
    VI_sim       : array
                   (if permutations>0)
                   variance of Is from permutations
    seI_sim      : array
                   (if permutations>0)
                   standard deviations of Is under permutations.
    z_sim        : arrray
                   (if permutations>0)
                   standardized Is based on permutations
    p_z_sim      : array
                   (if permutations>0)
                   p-values based on standard normal approximation from
                   permutations (one-sided)
                   for two-sided tests, these values should be multiplied by 2


    Examples
    --------
    >>> import pysal as ps
    >>> import numpy as np
    >>> np.random.seed(10)
    >>> w = ps.open(ps.examples.get_path("sids2.gal")).read()
    >>> f = ps.open(ps.examples.get_path("sids2.dbf"))
    >>> x = np.array(f.by_col['SIDR79'])
    >>> y = np.array(f.by_col['SIDR74'])
    >>> lm = ps.Moran_Local_BV(x, y, w, transformation = "r", \
                               permutations = 99)
    >>> lm.q[:10]
    array([3, 4, 3, 4, 2, 1, 4, 4, 2, 4])
    >>> lm.p_z_sim[0]
    0.0017240031348827456
    >>> lm = ps.Moran_Local_BV(x, y, w, transformation = "r", \
                               permutations = 99, geoda_quads=True)
    >>> lm.q[:10]
    array([2, 4, 2, 4, 3, 1, 4, 4, 3, 4])

    Note random components result is slightly different values across
    architectures so the results have been removed from doctests and will be
    moved into unittests that are conditional on architectures
    """
    def __init__(self, x, y, w, transformation="r", permutations=PERMUTATIONS,
                 geoda_quads=False):
        x = np.asarray(x).flatten()#将数组折叠成一维的数组
        y = np.asarray(y).flatten()
        self.y = y
        n = len(y)
        self.n = n
        self.n_1 = n - 1
        zx = x - x.mean()#标准化a的方式
        zy = y - y.mean()
        # setting for floating point noise
        orig_settings = np.seterr()
        np.seterr(all="ignore")
        sx = x.std()#果然还是用的标准化值。
        zx /= sx
        sy = y.std()
        zy /= sy
        np.seterr(**orig_settings)
        self.zx = zx
        self.zy = zy
        w.transform = transformation
        self.w = w
        self.permutations = permutations
        self.den = (zx * zx).sum()#这个是个什么鬼？
        self.Is = self.calc(self.w, self.zx, self.zy)
        self.geoda_quads = geoda_quads
        quads = [1, 2, 3, 4]
        if geoda_quads:
            quads = [1, 3, 2, 4]
        self.quads = quads
        self.__quads()
        if permutations:#这个if语句放在这里是啥意思？
            self.__crand()
            sim = np.transpose(self.rlisas)#使用了上面的方法，这个属性才能被调用,999行1519列的
            above = sim >= self.Is#这里也属于数组的广播，（999，1519）与（一个一维数组或者二维1*n或者列表才能进行广播，且这两个被比较的数组的列数需要相等），这一点一定要记住！！！！！！
            larger = above.sum(0)#求和了置换值大于推测值的数量，
            low_extreme = (self.permutations - larger) < larger
            larger[low_extreme] = self.permutations - larger[low_extreme]
            self.p_sim = (larger + 1.0) / (permutations + 1.0)
	    #这是这段代码最核心的一部分，p值和置换结果大与计算结果的数量有关。	
            self.sim = sim
            self.EI_sim = sim.mean(axis=0)
            self.seI_sim = sim.std(axis=0)
            self.VI_sim = self.seI_sim * self.seI_sim
            self.z_sim = (self.Is - self.EI_sim) / self.seI_sim
            self.p_z_sim = 1 - stats.norm.cdf(np.abs(self.z_sim))

    def calc(self, w, zx, zy):
        zly = slag(w, zy)
        return self.n_1 * self.zx * zly / self.den #这里计算的是个什么鬼？

    def __crand(self):
        """
        conditional randomization

        for observation i with ni neighbors,  the candidate set cannot include
        i (we don't want i being a neighbor of i). we have to sample without
        replacement from a set of ids that doesn't include i. numpy doesn't
        directly support sampling wo replacement and it is expensive to
        implement this. instead we omit i from the original ids,  permute the
        ids and take the first ni elements of the permuted ids as the
        neighbors to i in each randomization.

        """
        lisas = np.zeros((self.n, self.permutations))
        n_1 = self.n - 1  #将实体的总数减少1，然后获得数量3
        prange = range(self.permutations)
        k = self.w.max_neighbors + 1 #邻接的矩阵的总数加1
        nn = self.n - 1
        rids = np.array([np.random.permutation(nn)[0:k] for i in prange])
	#直接可以使用打印，然后打印出结果,形成了一个999行，15列的矩阵
        ids = np.arange(self.w.n)#形成一个0，1519的一维数组，
        ido = self.w.id_order#顺序得出每个unit的id号，把他们放在一个列表中
        w = [self.w.weights[ido[i]] for i in ids]#这里的weights是用来计算权重的，比如id为1的unit周边有八个相邻的实体，则每个实体所占比重为0.125，这个字典的键就是这个unit的id，值就是邻接unit的权重
	#上面这句代码的意思是，仅取出权重，那这样子，顺序不是被打乱了。self.w.weights 为一个字典
        wc = [self.w.cardinalities[ido[i]] for i in ids]##这个neighbor和cardinalies有点相似，前者的键是unit的id，值是和他成邻接关系的id，后者的键是unit的id 值是成邻接关系的id的数量。
	#上面这句代码最后会产生，包含数量关系的大列表，
        zx = self.zx
        zy = self.zy#引入标准化的zx与标准化后的zy
        for i in xrange(self.w.n):#xrange在python3中并没有，这里可以就把他们当成一个range来进行理解。
            idsi = ids[ids != i]#注意这个写法很有意思。把列表中和迭代序号一致的项去掉。由于ids是一个np.array项。这里实际上进行的numpy的数组索引,把false对应项直接去除掉。（这里其实也是一个花式索引）
            np.random.shuffle(idsi)#打乱数组
            tmp = zy[idsi[rids[:, 0:wc[i]]]] #rids[:, 0:wc[i]] 指的是取出一个999行，wc[i]列的随机数矩阵。然后用这个二维数组对一维数组进行花式索引，得到的一维数组的形状和rids的形状是一样的。
	    #这一句话里包含的意思还是很多的。idsi[rids[:, 0:wc[i]]]保证得到的999*wc[i]的数组中不包含i，zy[idsi[rids[:, 0:wc[i]]]]保证得到的数组中的999*wc[i]的数组的id与id对应的y轴标准化值对应起来。
	    #这个语句的打乱程度还是很高的，不光idsi在运算的时候被打乱了，rids在运算的时候也是个随机矩阵
            lisas[i] = zx[i] * (w[i] * tmp).sum(1)#w[i] * tmp标量与矢量相乘，数组的广播,相当于每行的结果乘以了一个权重， 结果是999行，wc[i]列的随机数矩阵，zx[i]相当于取出目标unit的x标准化值，
	    #上面这个语句记得要	
        self.rlisas = (n_1 / self.den) * lisas 
 
    def __quads(self):
        zl = slag(self.w, self.zy)#得出空间滞后值
        zp = self.zx > 0#并没有很完整的计算出标准差，这里就是计算一个正负号
        lp = zl > 0
        pp = zp * lp
        np = (1 - zp) * lp
        nn = (1 - zp) * (1 - lp)
        pn = zp * (1 - lp)
        self.q = self.quads[0] * pp + self.quads[1] * np + self.quads[2] * nn \
            + self.quads[3] * pn

    @property
    def _statistic(self):
        """More consistent hidden attribute to access ESDA statistics"""
        return self.Is

    @classmethod
    def by_col(cls, df, x, y=None, w=None, inplace=False, pvalue='sim', outvals=None, **stat_kws):
        """ 
        Function to compute a Moran_Local_BV statistic on a dataframe

        Arguments
        ---------
        df          :   pandas.DataFrame
                        a pandas dataframe with a geometry column
        X           :   list of strings
                        column name or list of column names to use as X values to compute
                        the bivariate statistic. If no Y is provided, pairwise comparisons
                        among these variates are used instead. 
        Y           :   list of strings
                        column name or list of column names to use as Y values to compute
                        the bivariate statistic. if no Y is provided, pariwise comparisons
                        among the X variates are used instead. 
        w           :   pysal weights object
                        a weights object aligned with the dataframe. If not provided, this
                        is searched for in the dataframe's metadata
        inplace     :   bool
                        a boolean denoting whether to operate on the dataframe inplace or to
                        return a series contaning the results of the computation. If
                        operating inplace, the derived columns will be named
                        'column_moran_local_bv'
        pvalue      :   string
                        a string denoting which pvalue should be returned. Refer to the
                        the Moran_Local_BV statistic's documentation for available p-values
        outvals     :   list of strings
                        list of arbitrary attributes to return as columns from the 
                        Moran_Local_BV statistic
        **stat_kws  :   keyword arguments
                        options to pass to the underlying statistic. For this, see the
                        documentation for the Moran_Local_BV statistic.


        Returns
        --------
        If inplace, None, and operation is conducted on dataframe in memory. Otherwise,
        returns a copy of the dataframe with the relevant columns attached.

        See Also
        ---------
        For further documentation, refer to the Moran_Local_BV class in pysal.esda
        """
        return _bivariate_handler(df, x, y=y, w=w, inplace=inplace, 
                                  pvalue = pvalue, outvals = outvals, 
                                  swapname=cls.__name__.lower(), stat=cls,**stat_kws)
