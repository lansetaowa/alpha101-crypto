from alpha_utils import *

def alpha001(c, r):
    """
    Alpha Factor #1: (rank(ts_argmax(power(((returns < 0) ? ts_std(returns, 20) : close), 2.), 5)) - 0.5)

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。索引为日期，列为股票代码。
        r: pandas DataFrame, 收益率。索引为日期，列为股票代码。
    适用性:
        截面 Alpha (使用了 rank 和 ts_argmax)
    Alpha 含义:
        此 Alpha 计算逻辑较为复杂。
        1. 如果收益率为负，则使用过去20天的收益率标准差替换对应的收盘价。
        2. 对处理后的价格（或标准差）取平方。
        3. 在过去5天的时间序列上，找到上述平方值的最大值所在的位置（ts_argmax）。
        4. 对这个位置进行横截面排名（rank）。
        5. 将排名结果减去0.5，并乘以-0.5。
        这个 Alpha 可能试图捕捉价格在经历波动（负收益时）或强势上涨后的短期反转或持续。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    c[r < 0] = ts_std(r, 20)
    return (rank(ts_argmax(power(c, 2), 5)).mul(-.5)
            .stack().swaplevel())

def alpha002(o, c, v):
    """
    Alpha Factor #2: (-1 * ts_corr(rank(ts_delta(log(volume), 2)), rank(((close - open) / open)), 6))

    中文注释:
    入参:
        o: pandas DataFrame, 开盘价。索引为日期，列为股票代码。
        c: pandas DataFrame, 收盘价。索引为日期，列为股票代码。
        v: pandas DataFrame, 成交量。索引为日期，列为股票代码。
    适用性:
        截面 Alpha (使用了 rank 和 ts_corr 进行横向比较)
    Alpha 含义:
        此 Alpha 计算成交量变化率的排名与日内收益率 ((close - open) / open) 排名在过去6天内的负相关性。
        1. 计算成交量在2天内的对数变化率，并对其进行横截面排名。
        2. 计算日内收益率 ((close - open) / open)，并对其进行横截面排名。
        3. 计算上述两个排名序列在过去6天的时间序列相关性。
        4. 取相关性的负值。
        该 Alpha 可能试图捕捉成交量变化与价格日内表现之间的反向关系。例如，成交量显著增加但日内价格表现不佳（或反之）的股票。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。包含 NaN 值（由 replace([-np.inf, np.inf], np.nan) 处理）。
    """
    s1 = rank(ts_delta(log(v), 2))
    s2 = rank((c / o) - 1)
    alpha = -ts_corr(s1, s2, 6)
    return alpha.stack('ticker').swaplevel().replace([-np.inf, np.inf], np.nan)

def alpha003(o, v):
    """
    Alpha Factor #3: (-1 * ts_corr(rank(open), rank(volume), 10))

    中文注释:
    入参:
        o: pandas DataFrame, 开盘价。索引为日期，列为股票代码。
        v: pandas DataFrame, 成交量。索引为日期，列为股票代码。
    适用性:
        截面 Alpha (使用了 rank 和 ts_corr)
    Alpha 含义:
        此 Alpha 计算开盘价排名和成交量排名在过去10天内的负相关性。
        1. 对开盘价进行横截面排名。
        2. 对成交量进行横截面排名。
        3. 计算上述两个排名序列在过去10天的时间序列相关性。
        4. 取相关性的负值。
        该 Alpha 可能试图捕捉开盘价和成交量之间的反向关系。例如，开盘价排名较高但成交量排名较低的股票。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。包含 NaN 值。
    """

    return (-ts_corr(rank(o), rank(v), 10)
            .stack('ticker')
            .swaplevel()
            .replace([-np.inf, np.inf], np.nan))

def alpha004(l):
    """
    Alpha Factor #4: (-1 * Ts_Rank(rank(low), 9))

    中文注释:
    入参:
        l: pandas DataFrame, 最低价。索引为日期，列为股票代码。
    适用性:
        截面 Alpha (主要依赖 rank 和 Ts_Rank/ts_rank)
    Alpha 含义:
        此 Alpha 计算最低价排名的时序排名的负值。
        1. 对每日最低价进行横截面排名 (rank(low))。
        2. 对上述排名结果，在过去9天的时间窗口内进行时序排名 (Ts_Rank 或 ts_rank)。
        3. 取最终结果的负值。
        该 Alpha 关注的是那些在近期（9天内）持续处于较低排名（相对于其他股票的最低价）的股票。负号可能表示对这类股票的看跌观点。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    return (-ts_rank(rank(l), 9)
            .stack('ticker')
            .swaplevel())

def alpha005(o, vwap, c):
    """
    Alpha Factor #5: (rank((open - ts_mean(vwap, 10))) * (-1 * abs(rank((close - vwap)))))

    中文注释:
    入参:
        o: pandas DataFrame, 开盘价。索引为日期，列为股票代码。
        vwap: pandas DataFrame, 成交量加权平均价。索引为日期，列为股票代码。
        c: pandas DataFrame, 收盘价。索引为日期，列为股票代码。
    适用性:
        截面 Alpha (使用了 rank)
    Alpha 含义:
        此 Alpha 结合了开盘价与10日VWAP均线的偏离，以及收盘价与当日VWAP的偏离。
        1. 计算开盘价与过去10日VWAP均值的差值，并对其进行横截面排名 (rank(open - ts_mean(vwap, 10)))。
        2. 计算收盘价与当日VWAP的差值，对其进行横截面排名，取绝对值，然后乘以-1 ( -1 * abs(rank(close - vwap)) )。
        3. 将上述两个结果相乘。
        该 Alpha 试图寻找那些开盘显著高于近期VWAP均值，并且收盘价相对于当日VWAP的偏离度（取负绝对值后）也表现出特定模式的股票。
        具体来说，它可能偏好：开盘价远高于10日VWAP均值（排名靠前），同时收盘价也显著偏离当日VWAP（rank的绝对值较大，乘以-1后变成一个较大的负数）的股票，最终结果为负向因子。
        或者开盘价远低于10日VWAP均值（排名靠后，为负），同时收盘价也显著偏离当日VWAP（rank的绝对值较大，乘以-1后变成一个较大的负数）的股票，最终结果为正向因子。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    return (rank(o.sub(ts_mean(vwap, 10)))
            .mul(rank(c.sub(vwap)).mul(-1).abs())
            .stack('ticker')
            .swaplevel())

def alpha006(o, v):
    """
    Alpha Factor #6: (-ts_corr(open, volume, 10))

    中文注释:
    入参:
        o: pandas DataFrame, 开盘价。索引为日期，列为股票代码。
        v: pandas DataFrame, 成交量。索引为日期，列为股票代码。
    适用性:
        时序 Alpha (主要基于 ts_corr，但通常 Alpha 池中的因子会经过截面处理)
        注意：虽然原始公式中没有 rank，但通常这类因子在实际应用中会结合排名等截面操作。
              如果严格按照公式，它是时序的。但从Alpha库的整体风格看，最终输出会stack并swaplevel，暗示了截面比较的意图。
              此处我们根据原始公式的直接操作判断为时序，但指出其在组合中可能作为截面因子使用。
    Alpha 含义:
        此 Alpha 计算开盘价和成交量在过去10天内的负相关性。
        与 Alpha003 的区别在于，这里直接使用原始值进行相关性计算，而不是使用排名。
        它试图捕捉那些开盘价上涨而成交量萎缩（或开盘价下跌而成交量放大）的模式，并赋予负的alpha值。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    return (-ts_corr(o, v, 10)
            .stack('ticker')
            .swaplevel())

def alpha007(c, v, adv20):
    """
    Alpha Factor #7: (adv20 < volume) ? ((-ts_rank(abs(ts_delta(close, 7)), 60)) * sign(ts_delta(close, 7))) : -1

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。索引为日期，列为股票代码。
        v: pandas DataFrame, 成交量。索引为日期，列为股票代码。
        adv20: pandas DataFrame, 过去20日平均成交量。索引为日期，列为股票代码。
    适用性:
        截面/时序混合 Alpha (ts_rank 是时序排名, sign(ts_delta) 是时序操作, 条件判断 adv20 < volume 是截面比较)
    Alpha 含义:
        这是一个条件型 Alpha。
        1. 条件判断：如果当日成交量 v 大于过去20日平均成交量 adv20。
        2. 如果条件为真：
            a. 计算收盘价在过去7日的变化量 ts_delta(close, 7)。
            b. 取该变化量的绝对值，并在过去60天内进行时序排名 ts_rank(abs(ts_delta(close, 7)), 60)。
            c. 将上述排名结果乘以7日收盘价变化量的符号 sign(ts_delta(close, 7))。
            d. 最后整体乘以 -1。
            这个逻辑试图在放量的情况下，捕捉价格大幅变动（无论方向）后的反转信号或特定模式。
            例如，如果价格在7天内上涨（sign>0），且变动幅度在过去60天排名靠前，则因子值为负。
        3. 如果条件为假（当日成交量未超过20日均量）：Alpha 值为 -1。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    
    delta7 = ts_delta(c, 7)
    return (-ts_rank(abs(delta7), 60)
            .mul(sign(delta7))
            .where(adv20<v, -1)
            .stack('ticker')
            .swaplevel())

def alpha008(o, r):
    """
    Alpha Factor #8: -rank(((ts_sum(open, 5) * ts_sum(returns, 5)) - ts_lag((ts_sum(open, 5) * ts_sum(returns, 5)),10)))

    中文注释:
    入参:
        o: pandas DataFrame, 开盘价。索引为日期，列为股票代码。
        r: pandas DataFrame, 收益率。索引为日期，列为股票代码。
    适用性:
        截面 Alpha (使用了 rank)
    Alpha 含义:
        此 Alpha 关注 (5日开盘价总和 * 5日收益率总和) 这个组合指标的变化。
        1. 计算过去5日开盘价的总和 ts_sum(open, 5)。
        2. 计算过去5日收益率的总和 ts_sum(returns, 5)。
        3. 将上述两者相乘: product = ts_sum(open, 5) * ts_sum(returns, 5)。
        4. 计算该 product 指标与其10天前的值的差值: product - ts_lag(product, 10)。
        5. 对这个差值进行横截面排名 (rank)。
        6. 取排名的负值。
        该 Alpha 试图捕捉那些“开盘价累积效应”与“收益率累积效应”的乘积发生显著变化（相对于10天前）的股票。
        负号表示对该变化排名靠前的股票给予负的alpha值。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    return (-(rank(((ts_sum(o, 5) * ts_sum(r, 5)) -
                       ts_lag((ts_sum(o, 5) * ts_sum(r, 5)), 10))))
           .stack('ticker')
            .swaplevel())

def alpha009(c):
    """
    Alpha Factor #9: (0 < ts_min(ts_delta(close, 1), 5)) ? ts_delta(close, 1) : ((ts_max(ts_delta(close, 1), 5) < 0) ? ts_delta(close, 1) : (-1 * ts_delta(close, 1)))

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。索引为日期，列为股票代码。
    适用性:
        时序 Alpha (主要基于 ts_min, ts_max, ts_delta 对单只股票的时间序列进行操作)
    Alpha 含义:
        这是一个基于近期价格变动方向的条件型 Alpha。
        令 close_diff = ts_delta(close, 1) (当日收盘价与昨日收盘价的差)。
        1. 如果过去5天内 close_diff 的最小值大于0 (即过去5天每天都在上涨): Alpha = close_diff (当日涨幅)。
        2. 否则，如果过去5天内 close_diff 的最大值小于0 (即过去5天每天都在下跌): Alpha = close_diff (当日跌幅)。
        3. 否则 (即过去5天内有涨有跌): Alpha = -1 * close_diff (当日涨跌幅的反转)。
        该 Alpha 试图在持续上涨或持续下跌的趋势中顺势而为，在震荡市中则采取反转策略。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    close_diff = ts_delta(c, 1)
    alpha = close_diff.where(ts_min(close_diff, 5) > 0,
                             close_diff.where(ts_max(close_diff, 5) < 0,
                                              -close_diff))
    return (alpha
            .stack('ticker')
            .swaplevel())

def alpha010(c):
    """
    Alpha Factor #10: rank(((0 < ts_min(ts_delta(close, 1), 4)) ? ts_delta(close, 1) : ((ts_max(ts_delta(close, 1), 4) < 0) ? ts_delta(close, 1) : (-1 * ts_delta(close, 1)))))

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。索引为日期，列为股票代码。
    适用性:
        截面 Alpha (最后对条件逻辑的结果进行了 rank 操作)
    Alpha 含义:
        与 Alpha009 类似，但时间窗口改为4天，并且最后对结果进行了横截面排名。
        令 close_diff = ts_delta(close, 1)。
        1. 如果过去4天内 close_diff 的最小值大于0: inner_alpha = close_diff。
        2. 否则，如果过去4天内 close_diff 的最大值小于0: inner_alpha = close_diff。
        3. 否则: inner_alpha = -1 * close_diff。
        4. Alpha = rank(inner_alpha)。
        该 Alpha 同样试图根据近期价格趋势的持续性来决定是顺势还是反转，并通过排名来比较不同股票间的这种效应强度。
        注意原始代码中 `close_diff.where(ts_min(close_diff, 4) > 0, close_diff.where(ts_min(close_diff, 4) > 0, -close_diff))` 第二个条件似乎笔误，应为 `ts_max(close_diff, 4) < 0`。假设按原始公式注释。如果按 Alpha009 的逻辑修正，则含义与009的排名版一致。
        【按代码实现注释】第二个条件 `ts_min(close_diff, 4) > 0` 重复，这意味着如果第一个条件不满足，第二个条件永远不满足，直接执行 `-close_diff`。
        所以实际逻辑是：如果过去4天每日上涨，则为当日涨幅；否则为当日涨幅的反转。然后对这个结果进行排名。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    close_diff = ts_delta(c, 1)
    alpha = close_diff.where(ts_min(close_diff, 4) > 0,
                             close_diff.where(ts_min(close_diff, 4) > 0, # 假设按代码注释，这里是ts_min
                                              -close_diff))

    return (rank(alpha)
            .stack('ticker')
            .swaplevel())

def alpha011(c, vwap, v):
    """
    Alpha Factor #11: (rank(ts_max((vwap - close), 3)) + rank(ts_min(vwap - close), 3)) * rank(ts_delta(volume, 3))

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。索引为日期，列为股票代码。
        vwap: pandas DataFrame, 成交量加权平均价。索引为日期，列为股票代码。
        v: pandas DataFrame, 成交量。索引为日期，列为股票代码。
    适用性:
        截面 Alpha (使用了 rank)
    Alpha 含义:
        此 Alpha 综合了价格与VWAP的偏离程度以及成交量的变化。
        1. 计算 (VWAP - 收盘价) 在过去3日的最大值，并对其进行横截面排名: rank(ts_max(vwap - close, 3))。这表示价格低于VWAP的最大幅度。
        2. 计算 (VWAP - 收盘价) 在过去3日的最小值，并对其进行横截面排名: rank(ts_min(vwap - close, 3))。这表示价格高于VWAP的最大幅度 (如果vwap-close为负)。
        3. 将上述两个排名相加。
        4. 计算成交量在过去3日的变化量，并对其进行横截面排名: rank(ts_delta(volume, 3))。
        5. 将步骤3的结果与步骤4的结果相乘。
        该 Alpha 可能试图捕捉那些价格相对VWAP有显著偏离（无论向上或向下），并且成交量也发生显著变化的股票。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    return (rank(ts_max(vwap.sub(c), 3))
            .add(rank(ts_min(vwap.sub(c), 3)))
            .mul(rank(ts_delta(v, 3)))
            .stack('ticker')
            .swaplevel())

def alpha012(v, c):
    """
    Alpha Factor #12: (sign(ts_delta(volume, 1)) * (-1 * ts_delta(close, 1)))

    中文注释:
    入参:
        v: pandas DataFrame, 成交量。索引为日期，列为股票代码。
        c: pandas DataFrame, 收盘价。索引为日期，列为股票代码。
    适用性:
        时序 Alpha (直接基于价格和成交量的日度变化)
    Alpha 含义:
        此 Alpha 描述了量价关系的一种模式：
        1. 计算当日成交量变化的方向: sign(ts_delta(volume, 1))。如果成交量增加，为1；减少为-1；不变为0。
        2. 计算当日收盘价变化的反方向: -1 * ts_delta(close, 1)。如果价格上涨，此项为负；价格下跌，此项为正。
        3. 将两者相乘。
        - 如果成交量增加 (sign=1) 且价格下跌 (delta_close<0, -delta_close>0)，则 Alpha 为正 (放量下跌)。
        - 如果成交量增加 (sign=1) 且价格上涨 (delta_close>0, -delta_close<0)，则 Alpha 为负 (放量上涨)。
        - 如果成交量减少 (sign=-1) 且价格下跌 (delta_close<0, -delta_close>0)，则 Alpha 为负 (缩量下跌)。
        - 如果成交量减少 (sign=-1) 且价格上涨 (delta_close>0, -delta_close<0)，则 Alpha 为正 (缩量上涨)。
        该 Alpha 倾向于看好“放量下跌”和“缩量上涨”的股票，看空“放量上涨”和“缩量下跌”的股票。这是一种反转或背离的逻辑。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    return (sign(ts_delta(v, 1)).mul(-ts_delta(c, 1))
            .stack('ticker')
            .swaplevel())

def alpha013(c, v):
    """
    Alpha Factor #13: -rank(ts_cov(rank(close), rank(volume), 5))

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。索引为日期，列为股票代码。
        v: pandas DataFrame, 成交量。索引为日期，列为股票代码。
    适用性:
        截面 Alpha (核心是 rank 和 ts_cov 后的 rank)
    Alpha 含义:
        此 Alpha 计算收盘价排名和成交量排名在过去5日内的协方差，然后对该协方差进行排名，并取负。
        1. 对每日收盘价进行横截面排名: rank(close)。
        2. 对每日成交量进行横截面排名: rank(volume)。
        3. 计算上述两个排名序列在过去5天的时间序列协方差: ts_cov(rank(close), rank(volume), 5)。
        4. 对这个协方差值进行横截面排名。
        5. 取最终排名的负值。
        该 Alpha 关注的是收盘价排名和成交量排名的同步变动性。
        如果两者排名同向变动（高收盘价排名对应高成交量排名），协方差为正，排名靠前，最终alpha为负。
        如果两者排名反向变动，协方差为负，排名靠后，最终alpha为正。
        它可能试图做空那些量价配合良好（排名）的股票，做多那些量价背离（排名）的股票。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    return (-rank(ts_cov(rank(c), rank(v), 5))
            .stack('ticker')
            .swaplevel())

def alpha014(o, v, r):
    """
    Alpha Factor #14: (-rank(ts_delta(returns, 3))) * ts_corr(open, volume, 10))

    中文注释:
    入参:
        o: pandas DataFrame, 开盘价。索引为日期，列为股票代码。
        v: pandas DataFrame, 成交量。索引为日期，列为股票代码。
        r: pandas DataFrame, 收益率。索引为日期，列为股票代码。
    适用性:
        截面 Alpha (使用了 rank 和 ts_corr)
    Alpha 含义:
        此 Alpha 结合了收益率的变化和开盘价与成交量的相关性。
        1. 计算收益率在过去3日的变化量，对其进行横截面排名，然后取负: -rank(ts_delta(returns, 3))。
           这表示近期收益率变化排名靠前的股票（例如，收益率加速上涨），此项为较大的负值。
        2. 计算开盘价和成交量在过去10天的时间序列相关性: ts_corr(open, volume, 10)。
        3. 将上述两项相乘。
        该 Alpha 试图寻找那些近期收益率变化显著（例如加速上涨或下跌），同时其开盘价与成交量表现出特定相关性模式的股票。
        例如，如果一只股票收益率加速上涨（第一项为负），并且其开盘价与成交量呈正相关（第二项为正），则最终 Alpha 为负。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。包含 NaN 值。
    """

    alpha = -rank(ts_delta(r, 3)).mul(ts_corr(o, v, 10)
                                      .replace([-np.inf,
                                                np.inf],
                                               np.nan))
    return (alpha
            .stack('ticker')
            .swaplevel())

def alpha015(h, v):
    """
    Alpha Factor #15: (-1 * ts_sum(rank(ts_corr(rank(high), rank(volume), 3)), 3))

    中文注释:
    入参:
        h: pandas DataFrame, 最高价。索引为日期，列为股票代码。
        v: pandas DataFrame, 成交量。索引为日期，列为股票代码。
    适用性:
        截面 Alpha (核心是 rank, ts_corr, ts_sum)
    Alpha 含义:
        此 Alpha 关注最高价排名与成交量排名的短期相关性的累积效应。
        1. 对每日最高价进行横截面排名: rank(high)。
        2. 对每日成交量进行横截面排名: rank(volume)。
        3. 计算上述两个排名序列在过去3天的时间序列相关性: ts_corr(rank(high), rank(volume), 3)。
        4. 对这个相关性值进行横截面排名。
        5. 将过去3天的这个排名值进行时间序列加总: ts_sum(..., 3)。
        6. 取最终结果的负值。
        该 Alpha 试图识别那些在过去几天内，最高价排名和成交量排名的相关性排名持续较高（或较低）的股票。
        负号表示对累积排名和较高的股票给予负的 Alpha 值。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。包含 NaN 值。
    """
    alpha = (-ts_sum(rank(ts_corr(rank(h), rank(v), 3)
                          .replace([-np.inf, np.inf], np.nan)), 3))
    return (alpha
            .stack('ticker')
            .swaplevel())

def alpha016(h, v):
    """
    Alpha Factor #16: (-1 * rank(ts_cov(rank(high), rank(volume), 5)))

    中文注释:
    入参:
        h: pandas DataFrame, 最高价。索引为日期，列为股票代码。
        v: pandas DataFrame, 成交量。索引为日期，列为股票代码。
    适用性:
        截面 Alpha (核心是 rank 和 ts_cov 后的 rank)
    Alpha 含义:
        与 Alpha013 类似，但用最高价 (high) 替代了收盘价 (close)。
        计算最高价排名和成交量排名在过去5日内的协方差，然后对该协方差进行排名，并取负。
        1. rank(high): 最高价的横截面排名。
        2. rank(volume): 成交量的横截面排名。
        3. ts_cov(rank(high), rank(volume), 5): 两者排名在过去5日的时序协方差。
        4. rank(...): 对上述协方差进行横截面排名。
        5. -1 * ...: 取负值。
        该 Alpha 关注最高价排名和成交量排名的同步变动性。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    return (-rank(ts_cov(rank(h), rank(v), 5))
            .stack('ticker')
            .swaplevel())

def alpha017(c, v):
    """
    Alpha Factor #17: (((-1 * rank(ts_rank(close, 10))) * rank(ts_delta(ts_delta(close, 1), 1))) *rank(ts_rank((volume / adv20), 5)))

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。索引为日期，列为股票代码。
        v: pandas DataFrame, 成交量。索引为日期，列为股票代码。
        (adv20 会在函数内基于 v 计算得到: adv20 = ts_mean(v, 20))
    适用性:
        截面 Alpha (大量使用了 rank 和 ts_rank)
    Alpha 含义:
        此 Alpha 结构复杂，结合了收盘价的趋势、加速度以及成交量的相对强度。
        1. part1 = -1 * rank(ts_rank(close, 10)):
           - ts_rank(close, 10): 收盘价在过去10天的时序排名。表示当前价格在近期所处的位置。
           - rank(...): 对上述时序排名进行横截面排名。
           - * -1: 取负。
           这部分关注那些近期价格时序排名靠前的股票（截面排名也靠前），并给予负向权重。
        2. part2 = rank(ts_delta(ts_delta(close, 1), 1)):
           - ts_delta(close, 1): 收盘价的日度变化（一阶差分）。
           - ts_delta(..., 1): 对日度变化再做一次差分，即价格的二阶差分（加速度）。
           - rank(...): 对价格加速度进行横截面排名。
           这部分关注价格变动加速的股票。
        3. part3 = rank(ts_rank((volume / adv20), 5)):
           - volume / adv20: 当日成交量相对于过去20日平均成交量的比率（量比）。
           - ts_rank(..., 5): 量比在过去5天的时序排名。
           - rank(...): 对上述时序排名进行横截面排名。
           这部分关注近期成交量相对强度（量比）持续较高的股票。
        最终 Alpha 是 part1 * part2 * part3。
        它试图寻找满足特定价格趋势、价格加速度和成交量强度组合条件的股票。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    adv20 = ts_mean(v, 20)
    return (-rank(ts_rank(c, 10))
            .mul(rank(ts_delta(ts_delta(c, 1), 1)))
            .mul(rank(ts_rank(v.div(adv20), 5)))
            .stack('ticker')
            .swaplevel())

def alpha018(o, c):
    """
    Alpha Factor #18: -rank((ts_std(abs((close - open)), 5) + (close - open)) + ts_corr(close, open,10))

    中文注释:
    入参:
        o: pandas DataFrame, 开盘价。索引为日期，列为股票代码。
        c: pandas DataFrame, 收盘价。索引为日期，列为股票代码。
    适用性:
        截面 Alpha (使用了 rank)
    Alpha 含义:
        此 Alpha 结合了日内价格波动、日内收益以及开盘价与收盘价的相关性。
        1. ts_std(abs(close - open), 5): 过去5日内，每日日内价格振幅（收盘价与开盘价差的绝对值）的标准差。表示近期日内波动稳定性。
        2. (close - open): 当日的日内收益。
        3. ts_corr(close, open, 10): 过去10日收盘价与开盘价的相关性。
        4. 将上述三项加总。
        5. 对总和进行横截面排名，并取负。
        该 Alpha 试图识别那些日内波动、日内收益和开盘收盘价联动性综合表现排名靠前的股票，并给予负的 Alpha 值。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。包含 NaN 值。
    """
    return (-rank(ts_std(c.sub(o).abs(), 5)
                  .add(c.sub(o))
                  .add(ts_corr(c, o, 10)
                       .replace([-np.inf,
                                 np.inf],
                                np.nan)))
            .stack('ticker')
            .swaplevel())

def alpha019(c, r):
    """
    Alpha Factor #19: ((-1 * sign(((close - ts_lag(close, 7)) + ts_delta(close, 7)))) * (1 + rank((1 + ts_sum(returns,250)))))

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。索引为日期，列为股票代码。
        r: pandas DataFrame, 收益率。索引为日期，列为股票代码。
    适用性:
        截面 Alpha (使用了 rank)
    Alpha 含义:
        此 Alpha 结合了短期价格趋势的加强信号和长期累积收益的排名。
        1. part1 = -1 * sign(((close - ts_lag(close, 7)) + ts_delta(close, 7))):
           - (close - ts_lag(close, 7)) 是过去7日的收盘价变化。
           - ts_delta(close, 7) 也是过去7日的收盘价变化。
           - 两者相加相当于 2 * ts_delta(close, 7)。
           - sign(...) 取其符号。
           - * -1: 取负。
           如果过去7日价格上涨，此项为-1；如果下跌，为1。
        2. part2 = (1 + rank((1 + ts_sum(returns, 250)))):
           - ts_sum(returns, 250): 过去250日（约一年）的累积收益率。
           - 1 + ...: 加1。
           - rank(...): 对其进行横截面排名。
           - 1 + ...: 排名结果再加1。
           这部分关注长期表现较好的股票（排名高）。
        最终 Alpha 是 part1 * part2。
        - 如果股票过去7日上涨（part1为-1），且长期收益排名靠前（part2为较大的正数），则Alpha为较大的负数。
        - 如果股票过去7日下跌（part1为1），且长期收益排名靠前（part2为较大的正数），则Alpha为较大的正数。
        该 Alpha 似乎倾向于做空短期上涨但长期强势的股票，做多短期下跌但长期强势的股票（一种均值回归的思路）。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    return (-sign(ts_delta(c, 7) + ts_delta(c, 7)) # (close - ts_lag(close, 7)) is equivalent to ts_delta(close, 7)
            .mul(1 + rank(1 + ts_sum(r, 250)))
            .stack('ticker')
            .swaplevel())

def alpha020(o, h, l, c):
    """
    Alpha Factor #20: -rank(open - ts_lag(high, 1)) * rank(open - ts_lag(close, 1)) * rank(open -ts_lag(low, 1))

    中文注释:
    入参:
        o: pandas DataFrame, 开盘价。索引为日期，列为股票代码。
        h: pandas DataFrame, 最高价。索引为日期，列为股票代码。
        l: pandas DataFrame, 最低价。索引为日期，列为股票代码。
        c: pandas DataFrame, 收盘价。索引为日期，列为股票代码。
    适用性:
        截面 Alpha (使用了 rank)
    Alpha 含义:
        此 Alpha 考察当日开盘价相对于昨日关键价位（最高、收盘、最低）的突破情况。
        1. rank(open - ts_lag(high, 1)): 当日开盘价与昨日最高价之差的排名。正值表示向上突破昨日高点。
        2. rank(open - ts_lag(close, 1)): 当日开盘价与昨日收盘价之差的排名。
        3. rank(open - ts_lag(low, 1)): 当日开盘价与昨日最低价之差的排名。
        4. 将这三个排名相乘，然后取负。
        该 Alpha 试图捕捉开盘跳空的行为。
        - 如果开盘价同时高于昨日最高、收盘、最低（三个排名都为正，乘积为正），最终Alpha为负。
        - 如果开盘价同时低于昨日最高、收盘、最低（三个排名都为负，乘积为负），最终Alpha也为负。
        - 其他情况则根据具体突破的组合决定。
        它对强势开盘（大幅高于昨日区间）或弱势开盘（大幅低于昨日区间）都给予负向信号，可能暗示反转。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    return (rank(o - ts_lag(h, 1))
            .mul(rank(o - ts_lag(c, 1)))
            .mul(rank(o - ts_lag(l, 1)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())

def alpha021(c, v):
    """
    Alpha Factor #21:
    ts_mean(close, 8) + ts_std(close, 8) < ts_mean(close, 2) ? -1
        : (ts_mean(close,2) < ts_mean(close, 8) - ts_std(close, 8) ? 1
            : (volume / adv20 < 1 ? -1 : 1))

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。索引为日期，列为股票代码。
        v: pandas DataFrame, 成交量。索引为日期，列为股票代码。
        (adv20 会在函数内基于 v 计算得到: adv20 = ts_mean(v, 20))
    适用性:
        时序 Alpha (基于布林带思路和量比进行条件判断)
    Alpha 含义:
        这是一个基于布林带概念和成交量过滤的条件型 Alpha。
        sma2 = 2日收盘价均线
        sma8 = 8日收盘价均线
        std8 = 8日收盘价标准差
        upper_band = sma8 + std8 (简化的布林上轨)
        lower_band = sma8 - std8 (简化的布林下轨)

        1. 条件1: 如果 upper_band < sma2 (即2日线突破上轨): Alpha = -1。
        2. 条件2: 否则，如果 sma2 < lower_band (即2日线跌破下轨): Alpha = 1。
        3. 条件3: 否则（即2日线在布林带之间），如果 volume / adv20 < 1 (当日缩量): Alpha = -1。
        4. 默认情况: Alpha = 1 (当日不缩量且2日线在布林带之间)。

        该 Alpha:
        - 对向上突破布林带的情况给 -1 (预期反转或回调)。
        - 对向下跌破布林带的情况给 1 (预期反弹)。
        - 在布林带内部，如果缩量给 -1，不缩量给 1。
    出参:
        pandas Series, 计算得到的 Alpha 值 (-1 或 1)。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    sma2 = ts_mean(c, 2)
    sma8 = ts_mean(c, 8)
    std8 = ts_std(c, 8)
    adv20 = ts_mean(v, 20) # adv20 is used in the original formula string

    cond_1 = sma8.add(std8) < sma2
    # The original formula string has: (ts_mean(close,2) < ts_mean(close, 8) - ts_std(close, 8)
    # My interpretation of cond_2 in code (sma8.add(std8) > sma2) is different from the formula string.
    # The code's cond_2: sma2 > sma8 + std8 which is NOT what's in the formula string.
    # Let's stick to the formula string for the conditions.
    cond_2_formula = sma2 < sma8.sub(std8) # sma2 < lower_band
    cond_3_formula = v.div(adv20) < 1

    # The np.select in the code uses cond_1, cond_2 (code's version), cond_3
    # choicelist=[-1, 1, -1], default=1
    # This means:
    # if cond_1: -1
    # else if cond_2 (code): 1
    # else if cond_3: -1
    # else: 1
    # To match the formula string:
    # if upper_band < sma2: -1
    # else if sma2 < lower_band: 1
    # else if volume / adv20 < 1: -1
    # else: 1

    alpha_values = np.select(
        condlist=[
            cond_1,             # sma8 + std8 < sma2
            cond_2_formula,     # sma2 < sma8 - std8
            cond_3_formula      # v / adv20 < 1
        ],
        choicelist=[-1, 1, -1],
        default=1
    )

    alpha = pd.DataFrame(alpha_values, index=c.index, columns=c.columns)

    return (alpha
            .stack('ticker')
            .swaplevel())

def alpha022(h, c, v):
    """
    Alpha Factor #22: -(ts_delta(ts_corr(high, volume, 5), 5) * rank(ts_std(close, 20)))

    中文注释:
    入参:
        h: pandas DataFrame, 最高价。索引为日期，列为股票代码。
        c: pandas DataFrame, 收盘价。索引为日期，列为股票代码。
        v: pandas DataFrame, 成交量。索引为日期，列为股票代码。
    适用性:
        截面 Alpha (包含 rank)
    Alpha 含义:
        此 Alpha 结合了最高价与成交量相关性的变化以及收盘价的波动率。
        1. ts_corr(high, volume, 5): 最高价与成交量在过去5日的时间序列相关性。
        2. ts_delta(..., 5): 上述相关性在过去5日的变化量。这表示量价相关性的变化速度。
        3. rank(ts_std(close, 20)): 过去20日收盘价标准差（波动率）的横截面排名。
        4. 将步骤2和步骤3的结果相乘，然后取负。
        该 Alpha 试图捕捉那些“最高价-成交量”相关性发生显著变化，并且当前波动率排名也较高的股票。
        负号使得：
        - 如果相关性增强 (delta > 0) 且波动率高 (rank > 0)，则 Alpha 为负。
        - 如果相关性减弱 (delta < 0) 且波动率高 (rank > 0)，则 Alpha 为正。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。包含 NaN 值。
    """

    return (ts_delta(ts_corr(h, v, 5)
                     .replace([-np.inf,
                               np.inf],
                              np.nan), 5)
            .mul(rank(ts_std(c, 20)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())

def alpha023(h, c): # Parameter c is not used in the formula or code
    """
    Alpha Factor #23: ((ts_mean(high, 20) < high) ? (-1 * ts_delta(high, 2)) : 0)

    中文注释:
    入参:
        h: pandas DataFrame, 最高价。索引为日期，列为股票代码。
        c: pandas DataFrame, 收盘价。索引为日期，列为股票代码。(注意：参数c在公式中未被使用)
    适用性:
        时序 Alpha (基于价格突破和价格变化)
    Alpha 含义:
        这是一个条件型 Alpha，关注最高价的突破行为。
        1. 条件: 如果当日最高价 high 大于过去20日的最高价均值 ts_mean(high, 20) (即向上突破近期高点区域)。
        2. 如果条件为真: Alpha = -1 * ts_delta(high, 2) (过去2日最高价变化量的负值)。
           这意味着如果近期（2日）最高价是上涨的 (ts_delta > 0)，则 Alpha 为负；如果近期最高价是下跌的 (ts_delta < 0)，则 Alpha 为正。
        3. 如果条件为假 (当日最高价未突破20日均线): Alpha = 0。
        该 Alpha 试图在最高价向上突破近期平均水平时，捕捉其短期动能的反转。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """

    return (ts_delta(h, 2)
            .mul(-1)
            .where(ts_mean(h, 20) < h, 0) # If ts_mean(high,20) < high is False, then result is 0
            .stack('ticker')
            .swaplevel())

def alpha024(c):
    """
    Alpha Factor #24: ((((ts_delta((ts_mean(close, 100)), 100) / ts_lag(close, 100)) <= 0.05) ? (-1 * (close - ts_min(close, 100))) : (-1 * ts_delta(close, 3))))

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。索引为日期，列为股票代码。
    适用性:
        时序 Alpha (基于长期趋势和短期价格行为的条件判断)
    Alpha 含义:
        这是一个条件型 Alpha，基于长期趋势的稳定性来选择不同的短期策略。
        1. 条件: (ts_delta(ts_mean(close, 100), 100) / ts_lag(close, 100)) <= 0.05
           - ts_mean(close, 100): 100日收盘价均线。
           - ts_delta(..., 100): 100日均线在过去100天的变化量。
           - ts_lag(close, 100): 100天前的收盘价。
           - 条件的含义是：100日均线的100日变动幅度相对于100天前的收盘价不超过5%。这表示一个相对平稳或缓慢变动的长期趋势。
        2. 如果条件为真 (长期趋势稳定): Alpha = -1 * (close - ts_min(close, 100))。
           - (close - ts_min(close, 100)) 表示当前价格相对于过去100日最低价的回撤幅度或向上偏离幅度。
           - 乘以 -1，意味着如果价格接近100日内低点，Alpha 接近0或为正；如果价格远高于100日内低点，Alpha 为负。这是一种在稳定趋势中的反转或区间交易逻辑。
        3. 如果条件为假 (长期趋势不稳定或变化较快): Alpha = -1 * ts_delta(close, 3)。
           - ts_delta(close, 3) 是过去3日的收盘价变化。
           - 乘以 -1，意味着如果价格在过去3日上涨，Alpha 为负；如果下跌，Alpha 为正。这是一种短期反转策略。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    cond = ts_delta(ts_mean(c, 100), 100) / ts_lag(c, 100) <= 0.05

    return (c.sub(ts_min(c, 100)).mul(-1).where(cond, -ts_delta(c, 3))
            .stack('ticker')
            .swaplevel())

def alpha025(h, c, r, vwap, adv20):
    """
    Alpha Factor #25: rank((-1 * returns) * adv20 * vwap * (high - close))

    中文注释:
    入参:
        h: pandas DataFrame, 最高价。索引为日期，列为股票代码。
        c: pandas DataFrame, 收盘价。索引为日期，列为股票代码。
        r: pandas DataFrame, 收益率。索引为日期，列为股票代码。
        vwap: pandas DataFrame, 成交量加权平均价。索引为日期，列为股票代码。
        adv20: pandas DataFrame, 过去20日平均成交量。索引为日期，列为股票代码。
    适用性:
        截面 Alpha (使用了 rank)
    Alpha 含义:
        此 Alpha 综合了多个因素，并对它们的乘积进行排名。
        1. -1 * returns: 当日收益率的负值。
        2. adv20: 20日平均成交量。
        3. vwap: 当日成交量加权平均价。
        4. (high - close): 当日最高价与收盘价之差（上影线长度，如果 h>c）。
        5. 将上述四项相乘: product = (-1 * r) * adv20 * vwap * (h - c)。
        6. rank(product): 对这个乘积进行横截面排名。

        该 Alpha 试图捕捉特定市场状态下的股票。例如：
        - 如果当日下跌 (r < 0, -r > 0)，且平均成交量大，VWAP价格高，同时有较长的上影线 (h-c > 0)，则 product 可能为较大的正数，排名靠前。
        - 它可能偏好那些在下跌日（或上涨但-r为正，即r为负）成交活跃（adv20, vwap），且日内从高点回落（h-c > 0）的股票。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    return (rank(-r.mul(adv20)
                 .mul(vwap)
                 .mul(h.sub(c)))
            .stack('ticker')
            .swaplevel())

def alpha026(h, v):
    """
    Alpha Factor #26: (-1 * ts_max(ts_corr(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))

    中文注释:
    入参:
        h: pandas DataFrame, 最高价。索引为日期，列为股票代码。
        v: pandas DataFrame, 成交量。索引为日期，列为股票代码。
    适用性:
        截面/时序混合 Alpha (ts_rank, ts_corr, ts_max)
    Alpha 含义:
        此 Alpha 关注成交量时序排名和最高价时序排名的相关性的近期最大值。
        1. ts_rank(volume, 5): 成交量在过去5日的时序排名。
        2. ts_rank(high, 5): 最高价在过去5日的时序排名。
        3. ts_corr(..., ..., 5): 上述两个时序排名在过去5日的相关性。
        4. ts_max(..., 3): 上述相关性在过去3日的最大值。
        5. 乘以 -1。
        该 Alpha 试图识别那些近期“成交量时序排名”与“最高价时序排名”表现出高度相关性（取其3日内最大值）的股票，并给予负的 Alpha 值。
        例如，如果一只股票的成交量和最高价在近期都处于其各自5日窗口的高位（时序排名靠前），并且这种正相关性在过去3天内达到了一个高点，则Alpha为负。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。包含 NaN 值。
    """
    return (ts_max(ts_corr(ts_rank(v, 5), 
                           ts_rank(h, 5), 5)
                   .replace([-np.inf, np.inf], np.nan), 3)
            .mul(-1)
            .stack('ticker')
            .swaplevel())

def alpha027(v, vwap):
    """
    Alpha Factor #27: ((0.5 < rank(ts_mean(ts_corr(rank(volume), rank(vwap), 6), 2))) ? -1 : 1)

    中文注释:
    入参:
        v: pandas DataFrame, 成交量。索引为日期，列为股票代码。
        vwap: pandas DataFrame, 成交量加权平均价。索引为日期，列为股票代码。
    适用性:
        截面 Alpha (核心是 rank 和条件判断)
    Alpha 含义:
        这是一个条件型 Alpha，基于成交量排名和VWAP排名的相关性的均值排名。
        1. rank(volume): 成交量的横截面排名。
        2. rank(vwap): VWAP的横截面排名。
        3. ts_corr(..., ..., 6): 上述两个排名在过去6日的时间序列相关性。
        4. ts_mean(..., 2): 上述相关性在过去2日的均值。
        5. rank(...): 对这个均值进行横截面排名。
        6. 条件判断: 如果步骤5的排名大于0.5 (即排名在前50%)，则 Alpha = -1。
        7. 否则: Alpha = 1。
        该 Alpha 试图识别那些“成交量排名与VWAP排名相关性的2日均值”在所有股票中排名靠前的股票，并给予 -1，否则给 1。
        它关注的是量价排名关系（通过VWAP体现）的持续性。
    出参:
        pandas Series, 计算得到的 Alpha 值 (-1 或 1)。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    cond_rank = rank(ts_mean(ts_corr(rank(v), rank(vwap), 6), 2))
    # Original code: alpha = cond.notnull().astype(float)
    # alpha.where(cond <= 0.5, -alpha)
    # This means if cond <= 0.5, result is 1*alpha (which is 1 if notnull, nan if null)
    # if cond > 0.5, result is -1*alpha (which is -1 if notnull, nan if null)
    # This is equivalent to: if cond > 0.5 then -1, else 1 (assuming notnull)
    alpha = pd.DataFrame(np.ones_like(cond_rank.values), index=cond_rank.index, columns=cond_rank.columns)
    alpha[cond_rank.isnull()] = np.nan
    alpha[cond_rank > 0.5] = -1

    return (alpha
            .stack('ticker')
            .swaplevel())

def alpha028(h, l, c, v, adv20):
    """
    Alpha Factor #28: scale(((ts_corr(adv20, low, 5) + (high + low) / 2) - close))

    中文注释:
    入参:
        h: pandas DataFrame, 最高价。索引为日期，列为股票代码。
        l: pandas DataFrame, 最低价。索引为日期，列为股票代码。
        c: pandas DataFrame, 收盘价。索引为日期，列为股票代码。
        v: pandas DataFrame, 成交量。索引为日期，列为股票代码。(用于计算 adv20)
        adv20: pandas DataFrame, 过去20日平均成交量。索引为日期，列为股票代码。
    适用性:
        截面 Alpha (使用了 scale 标准化)
    Alpha 含义:
        此 Alpha 构建了一个综合指标，并对其进行横截面标准化 (scale)。
        1. ts_corr(adv20, low, 5): 过去20日平均成交量 (adv20) 与最低价 (low) 在过去5日的时间序列相关性。
           (注意：代码中 adv20 是直接传入的，如果未传入则需要从v计算)
        2. (high + low) / 2: 当日最高价和最低价的均值，即当日中间价。
        3. 将步骤1和步骤2的结果相加，然后减去当日收盘价 close。
           inner_value = ts_corr(adv20, low, 5) + (high + low) / 2 - close
        4. scale(inner_value): 对上述计算结果进行横截面标准化（使其均值为0，标准差为1）。
        该 Alpha 结合了成交量与价格（最低价）的相关性、当日价格中枢以及收盘价的位置。
        标准化使得因子值在不同股票间具有可比性。
    出参:
        pandas Series, 计算得到的标准化 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    return (scale(ts_corr(adv20, l, 5)
                  .replace([-np.inf, np.inf], 0) # Handle potential inf from corr
                  .add(h.add(l).div(2).sub(c)))
            .stack('ticker')
            .swaplevel())

def alpha029(c, r):
    """
    Alpha Factor #29: (ts_min(ts_product(rank(rank(scale(log(ts_sum(ts_min(rank(rank((-1 * rank(ts_delta((close - 1),5))))), 2), 1))))), 1), 5) + ts_rank(ts_lag((-1 * returns), 6), 5))

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。索引为日期，列为股票代码。
        r: pandas DataFrame, 收益率。索引为日期，列为股票代码。
    适用性:
        截面 Alpha (大量使用了 rank, scale, ts_rank)
    Alpha 含义:
        此 Alpha 结构非常复杂，深度嵌套了多种运算，试图从价格和收益中提取复杂模式。
        Part 1: ts_min(ts_product(rank(rank(scale(log(ts_sum(ts_min(rank(rank((-1 * rank(ts_delta((close - 1),5))))), 2), 1))))), 1), 5)
            1. ts_delta(close - 1, 5): (收盘价-1)的5日变化。 (close-1) 可能是为了避免0或负值，或某种归一化。
            2. rank(...): 对其排名。
            3. -1 * ...: 取负。
            4. rank(...): 再排名。
            5. rank(...): 再排名。
            6. ts_min(..., 2): 取过去2日的最小值。
            7. ts_sum(..., 1): 实际上就是取当前值 (窗口为1的sum)。
            8. log(...): 取对数。
            9. scale(...): 标准化。
            10. rank(...): 排名。
            11. rank(...): 再排名。
            12. ts_product(..., 1): 实际上就是取当前值 (窗口为1的product)。
            13. ts_min(..., 5): 取过去5日的最小值。
        Part 2: ts_rank(ts_lag((-1 * returns), 6), 5)
            1. -1 * returns: 每日收益率的负值。
            2. ts_lag(..., 6): 滞后6天的负收益率。
            3. ts_rank(..., 5): 上述滞后负收益率在过去5天的时序排名。
        最终 Alpha = Part 1 + Part 2。
        由于其极端复杂性，很难给出一个简洁直观的经济学解释。它可能是在挖掘价格序列经过多层非线性变换后呈现的某种短期持续性或反转特性，并结合了滞后负收益率的近期表现。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    # The formula (close - 1) is unusual. Typically it would be just 'close'.
    # Let's assume 'close - 1' is intended.
    part1 = ts_min(ts_product(rank(rank(scale(log(ts_sum(ts_min(rank(rank(-rank(ts_delta(c.sub(1), 5)))), 2), 1))))), 1), 5)
    part2 = ts_rank(ts_lag(r.mul(-1), 6), 5)

    return (part1.add(part2)
            .stack('ticker')
            .swaplevel())

def alpha030(c, v):
    """
    Alpha Factor #30: (((1.0 - rank(((sign((close - ts_lag(close, 1))) + sign((ts_lag(close, 1) - ts_lag(close, 2)))) + sign((ts_lag(close, 2) - ts_lag(close, 3)))))) * ts_sum(volume, 5)) / ts_sum(volume, 20))

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。索引为日期，列为股票代码。
        v: pandas DataFrame, 成交量。索引为日期，列为股票代码。
    适用性:
        截面 Alpha (使用了 rank)
    Alpha 含义:
        此 Alpha 结合了近期价格变动方向的一致性与成交量的短期/长期比率。
        1. sign_sum = sign(ts_delta(close,1)) + sign(ts_lag(ts_delta(close,1),1)) + sign(ts_lag(ts_delta(close,1),2))
           这计算了最近三天每日价格变动方向的符号之和。
           - 如果连续三天上涨，sum = 3。
           - 如果连续三天下跌，sum = -3。
           - 其他情况介于 -3 和 3 之间。
        2. rank_val = rank(sign_sum)。 对上述符号和进行横截面排名。
        3. (1.0 - rank_val): 对排名取反向。如果符号和排名高（趋势强），则此项小。
        4. vol_ratio = ts_sum(volume, 5) / ts_sum(volume, 20): 5日成交量总和与20日成交量总和的比率。表示短期成交活跃度。
        5. Alpha = (1.0 - rank_val) * vol_ratio。

        该 Alpha 试图寻找：
        - 价格趋势不明显或反转（rank_val 较低，则 1-rank_val 较大）
        - 且短期成交量相对长期成交量较为活跃（vol_ratio 较大）
        的股票。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    close_diff = ts_delta(c, 1)
    return (rank(sign(close_diff)
                 .add(sign(ts_lag(close_diff, 1)))
                 .add(sign(ts_lag(close_diff, 2))))
            .mul(-1).add(1) # This is (1 - rank)
            .mul(ts_sum(v, 5))
            .div(ts_sum(v, 20))
            .stack('ticker')
            .swaplevel())

def alpha031(l, c, adv20):
    """
    Alpha Factor #31: ((rank(rank(rank(ts_weighted_mean((-1 * rank(rank(ts_delta(close, 10)))), 10)))) + rank((-1 * ts_delta(close, 3)))) + sign(scale(ts_corr(adv20, low, 12))))

    中文注释:
    入参:
        l: pandas DataFrame, 最低价。索引为日期，列为股票代码。
        c: pandas DataFrame, 收盘价。索引为日期，列为股票代码。
        adv20: pandas DataFrame, 过去20日平均成交量。索引为日期，列为股票代码。
    适用性:
        截面 Alpha (大量使用了 rank, scale, sign)
    Alpha 含义:
        此 Alpha 由三部分相加构成，结构复杂。
        Part 1: rank(rank(rank(ts_weighted_mean(-1 * rank(rank(ts_delta(close, 10))), 10))))
            - ts_delta(close, 10): 10日收盘价变化。
            - rank(rank(...)): 对其进行两层横截面排名。
            - -1 * ...: 取负。
            - ts_weighted_mean(..., 10): 对上述结果进行10日加权移动平均。
            - rank(rank(rank(...)))): 对加权平均结果进行三层横截面排名。
            这部分捕捉了价格变化经过多层排名和加权平均后的某种趋势强度。
        Part 2: rank(-1 * ts_delta(close, 3))
            - ts_delta(close, 3): 3日收盘价变化。
            - -1 * ...: 取负，表示短期反转。
            - rank(...): 对短期反转信号进行横截面排名。
        Part 3: sign(scale(ts_corr(adv20, low, 12)))
            - ts_corr(adv20, low, 12): 20日均量与最低价在过去12日的相关性。
            - scale(...): 对相关性进行横截面标准化。
            - sign(...): 取标准化后相关性的符号。
            这部分表示成交量与最低价相关性的方向。
        最终 Alpha = Part 1 + Part 2 + Part 3。
        它结合了中长期价格动能的复杂变换、短期价格反转以及量价相关性方向。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。包含 NaN 值。
    """
    return (rank(rank(rank(ts_weighted_mean(rank(rank(ts_delta(c, 10))).mul(-1), 10))))
            .add(rank(ts_delta(c, 3).mul(-1)))
            .add(sign(scale(ts_corr(adv20, l, 12)
                            .replace([-np.inf, np.inf],
                                     np.nan))))
            .stack('ticker')
            .swaplevel())

def alpha032(c, vwap):
    """
    Alpha Factor #32: scale(ts_mean(close, 7) - close) + (20 * scale(ts_corr(vwap, ts_lag(close, 5),230)))

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。索引为日期，列为股票代码。
        vwap: pandas DataFrame, 成交量加权平均价。索引为日期，列为股票代码。
    适用性:
        截面 Alpha (使用了 scale)
    Alpha 含义:
        此 Alpha 由两部分相加构成。
        Part 1: scale(ts_mean(close, 7) - close)
            - ts_mean(close, 7) - close: 7日均线与当日收盘价的差值。表示短期价格偏离均线的程度（均值回归信号）。
            - scale(...): 对其进行横截面标准化。
        Part 2: 20 * scale(ts_corr(vwap, ts_lag(close, 5), 230))
            - ts_lag(close, 5): 5日前收盘价。
            - ts_corr(vwap, ..., 230): 当日VWAP与5日前收盘价在过去230日（约一年）的相关性。这衡量了当前VWAP与稍早前价格的长期联动性。
            - scale(...): 对其进行横截面标准化。
            - 20 * ...: 将标准化后的结果乘以20放大。
        最终 Alpha = Part 1 + Part 2。
        它结合了短期均值回归信号和VWAP与滞后价格的长期相关性。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    return (scale(ts_mean(c, 7).sub(c))
            .add(20 * scale(ts_corr(vwap,
                                    ts_lag(c, 5), 230)))
            .stack('ticker')
            .swaplevel())

def alpha033(o, c):
    """
    Alpha Factor #33: rank(-(1 - (open / close)))

    中文注释:
    入参:
        o: pandas DataFrame, 开盘价。索引为日期，列为股票代码。
        c: pandas DataFrame, 收盘价。索引为日期，列为股票代码。
    适用性:
        截面 Alpha (使用了 rank)
    Alpha 含义:
        此 Alpha 基于开盘价与收盘价的比率。
        1. open / close: 开盘价除以收盘价。
        2. 1 - (open / close): 如果开盘价低于收盘价（当日上涨），此值为正；如果开盘价高于收盘价（当日下跌），此值为负。其大小反映了涨跌幅度相对于收盘价的比例。
        3. -(...) : 取负。当日上涨时，此值为负；当日下跌时，此值为正。
        4. rank(...): 对上述结果进行横截面排名。
        该 Alpha 倾向于给当日大幅下跌（相对于开盘价）的股票更高的排名，给大幅上涨的股票更低的排名。
        它实际上是在寻找日内反转的信号，或者说，对日内趋势的负向反应。
        等价于 rank((open / close) - 1) 或 rank(open/close)。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    # -(1 - (o/c)) = o/c - 1
    return (rank(o.div(c).mul(-1).add(1).mul(-1)) # This simplifies to rank(o.div(c) - 1)
            .stack('ticker')
            .swaplevel())

def alpha034(c, r):
    """
    Alpha Factor #34: rank(((1 - rank((ts_std(returns, 2) / ts_std(returns, 5)))) + (1 - rank(ts_delta(close, 1)))))

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。索引为日期，列为股票代码。
        r: pandas DataFrame, 收益率。索引为日期，列为股票代码。
    适用性:
        截面 Alpha (使用了 rank)
    Alpha 含义:
        此 Alpha 结合了短期波动率的变化和短期价格动量。
        Part 1: 1 - rank(ts_std(returns, 2) / ts_std(returns, 5))
            - ts_std(returns, 2): 2日收益率标准差 (短期波动)。
            - ts_std(returns, 5): 5日收益率标准差 (稍长期波动)。
            - ratio = ts_std(r, 2) / ts_std(r, 5): 短期波动与中期波动的比率。比率高表示近期波动放大。
            - rank(ratio): 对该比率进行横截面排名。
            - 1 - rank(ratio): 对波动放大排名靠前的股票给予较低的值。
        Part 2: 1 - rank(ts_delta(close, 1))
            - ts_delta(close, 1): 昨日到今日的收盘价变化 (日度动量)。
            - rank(...): 对日度动量进行横截面排名。
            - 1 - rank(...): 对日度动量排名靠前的股票（上涨）给予较低的值。
        Alpha = rank(Part 1 + Part 2)。 (注意：原公式是 rank( (1-rank_ratio) + (1-rank_delta) )，代码实现是 rank(rank_ratio * -1 - rank_delta + 2)，两者不等价，但最终都会再rank。这里按代码实现逻辑注释)
        代码实现逻辑: rank( (1-rank_ratio) + (1-rank_delta) )
        该 Alpha 试图寻找那些短期波动率相对稳定（或下降），并且短期价格动能也较弱（或下跌）的股票，并对这些股票的组合指标进行最终排名。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。包含 NaN 值。
    """
    # Formula interpretation: rank ( (1 - rank(std2/std5)) + (1 - rank(delta_close_1)) )
    # Code: rank( rank(std2/std5).mul(-1).sub(rank(delta_close_1)).add(2) )
    # rank(std2/std5).mul(-1) is like (const - rank(std2/std5))
    # .sub(rank(delta_close_1)) is like (const - rank(std2/std5) - rank(delta_close_1))
    # .add(2) is like (const' - rank(std2/std5) - rank(delta_close_1))
    # This is equivalent to rank( (k1 - rank(ratio)) + (k2 - rank(delta)) ) if k1+k2 = 2
    # So it is effectively rank( (1-rank(ratio)) + (1-rank(delta)) )
    return (rank(rank(ts_std(r, 2).div(ts_std(r, 5))
                      .replace([-np.inf, np.inf],
                               np.nan)) # inner rank 1: rank of ratio
                 .mul(-1) # (const - rank of ratio)
                 .add(1) # (1 - rank of ratio)
                 .add(rank(ts_delta(c, 1)).mul(-1).add(1)) # + (1 - rank of delta)
                ) # outer rank
            .stack('ticker')
            .swaplevel())

def alpha035(h, l, c, v, r):
    """
    Alpha Factor #35: ((ts_Rank(volume, 32) * (1 - ts_Rank(((close + high) - low), 16))) * (1 -ts_Rank(returns, 32)))

    中文注释:
    入参:
        h: pandas DataFrame, 最高价。索引为日期，列为股票代码。
        l: pandas DataFrame, 最低价。索引为日期，列为股票代码。
        c: pandas DataFrame, 收盘价。索引为日期，列为股票代码。
        v: pandas DataFrame, 成交量。索引为日期，列为股票代码。
        r: pandas DataFrame, 收益率。索引为日期，列为股票代码。
    适用性:
        时序 Alpha (主要基于 ts_Rank)
    Alpha 含义:
        此 Alpha 结合了成交量、价格位置和收益率的长期时序排名。
        1. ts_Rank(volume, 32): 成交量在过去32日的时序排名。排名高表示近期成交活跃。
        2. price_pos = (close + high) - low: 一个衡量价格位置的指标，结合了收盘价、最高价和最低价。
           ts_Rank(price_pos, 16): 上述价格位置指标在过去16日的时序排名。
           (1 - ts_Rank(price_pos, 16)): 对价格位置排名取反。如果价格位置指标的排名高，此项小。
        3. ts_Rank(returns, 32): 收益率在过去32日的时序排名。
           (1 - ts_Rank(returns, 32)): 对收益率排名取反。如果收益率排名高，此项小。
        Alpha = part1 * part2 * part3。
        该 Alpha 试图寻找那些：
        - 近期成交量时序排名高 (part1 大)
        - 且价格位置指标的近期时序排名低 (part2 大)
        - 且收益率的近期时序排名低 (part3 大)
        的股票。
        它可能在寻找成交活跃但价格和收益表现不佳（时序排名靠后，导致1-rank较大）的股票。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    return (ts_rank(v, 32)
            .mul(1 - ts_rank(c.add(h).sub(l), 16))
            .mul(1 - ts_rank(r, 32))
            .stack('ticker')
            .swaplevel())

def alpha036(o, c, v, r, adv20, vwap):
    """
    Alpha Factor #36: 2.21 * rank(ts_corr((close - open), ts_lag(volume, 1), 15)) + 0.7 * rank((open- close)) + 0.73 * rank(ts_Rank(ts_lag(-1 * returns, 6), 5)) + rank(abs(ts_corr(vwap,adv20, 6))) + 0.6 * rank(((ts_mean(close, 200) - open) * (close - open)))

    中文注释:
    入参:
        o: pandas DataFrame, 开盘价。
        c: pandas DataFrame, 收盘价。
        v: pandas DataFrame, 成交量。
        r: pandas DataFrame, 收益率。
        adv20: pandas DataFrame, 20日平均成交量。
        vwap: pandas DataFrame, 成交量加权平均价。
    适用性:
        截面 Alpha (各项均为 rank 或 rank of ts_rank)
    Alpha 含义:
        这是一个由多个加权项组成的复合 Alpha，各项含义如下：
        1. 2.21 * rank(ts_corr(close - open, ts_lag(volume, 1), 15)):
           日内收益 (c-o) 与滞后1日成交量在过去15日的相关性的排名。
        2. 0.7 * rank(open - close):
           日内收益的负值 (o-c) 的排名。关注日内下跌的股票。
        3. 0.73 * rank(ts_Rank(ts_lag(-1 * returns, 6), 5)):
           滞后6日的负收益率在过去5日的时序排名的横截面排名。关注近期（相对于6天前）表现不佳的股票。
        4. rank(abs(ts_corr(vwap, adv20, 6))):
           VWAP与20日均量在过去6日的相关性的绝对值的排名。关注VWAP与均量联动性强的股票（无论正负相关）。
        5. 0.6 * rank((ts_mean(close, 200) - open) * (close - open)):
           (200日均线 - 开盘价) * (日内收益) 的排名。
           如果日内上涨 (c-o > 0)，则偏好开盘价低于200日线的股票。
           如果日内下跌 (c-o < 0)，则偏好开盘价高于200日线的股票。
        该 Alpha 综合了多种短期量价模式、长期趋势与日内行为的交互。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """

    return (rank(ts_corr(c.sub(o), ts_lag(v, 1), 15)).mul(2.21)
            .add(rank(o.sub(c)).mul(.7))
            .add(rank(ts_rank(ts_lag(r.mul(-1), 6), 5)).mul(0.73)) # Corrected -r
            .add(rank(abs(ts_corr(vwap, adv20, 6))))
            .add(rank(ts_mean(c, 200).sub(o).mul(c.sub(o))).mul(0.6))
            .stack('ticker')
            .swaplevel())

def alpha037(o, c):
    """
    Alpha Factor #37: (rank(ts_corr(ts_lag((open - close), 1), close, 200)) + rank((open - close)))

    中文注释:
    入参:
        o: pandas DataFrame, 开盘价。
        c: pandas DataFrame, 收盘价。
    适用性:
        截面 Alpha (使用了 rank)
    Alpha 含义:
        此 Alpha 结合了滞后日内收益与当前收盘价的长期相关性，以及当日的日内收益。
        1. rank(ts_corr(ts_lag(open - close, 1), close, 200)):
           - ts_lag(open - close, 1): 昨日的日内收益的负值 (或昨日开盘价-昨日收盘价)。
           - ts_corr(..., close, 200): 上述指标与今日收盘价在过去200天的相关性。
           - rank(...): 对此相关性进行横截面排名。
        2. rank(open - close):
           - open - close: 当日的日内收益的负值。
           - rank(...): 对此进行横截面排名。
        最终 Alpha 是这两项排名之和。
        它试图寻找那些“昨日日内表现”与“今日收盘价”长期相关性较高，并且当日日内也表现出类似（负向）模式的股票。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    return (rank(ts_corr(ts_lag(o.sub(c), 1), c, 200))
            .add(rank(o.sub(c)))
            .stack('ticker')
            .swaplevel())

def alpha038(o, c): # Original formula string has close, but code uses open for ts_rank
    """
    Alpha Factor #38: "-1 * rank(ts_rank(close, 10)) * rank(close / open)"
    (Code uses open for ts_rank: -1 * rank(ts_rank(open, 10)) * rank(close / open) )

    中文注释: (按代码实现注释)
    入参:
        o: pandas DataFrame, 开盘价。
        c: pandas DataFrame, 收盘价。
    适用性:
        截面 Alpha (使用了 rank 和 ts_rank)
    Alpha 含义:
        此 Alpha 结合了开盘价的近期时序排名和收盘价与开盘价的比率的排名。
        1. rank(ts_rank(open, 10)):
           - ts_rank(open, 10): 开盘价在过去10日的时序排名。
           - rank(...): 对此时序排名进行横截面排名。
        2. rank(close / open):
           - close / open: 收盘价与开盘价的比率。大于1表示上涨，小于1表示下跌。
           - rank(...): 对此比率进行横截面排名。上涨的股票排名靠前。
        最终 Alpha = -1 * part1 * part2。
        - 如果一只股票近期开盘价时序排名高 (part1的rank值大)，且当日上涨 (part2的rank值大)，则乘积为大正数，最终Alpha为大负数。
        - 它倾向于做空那些“近期开盘强势”且“当日高开高走”的股票。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。包含 NaN 值。
    """
    return (rank(ts_rank(o, 10)) # Code uses open here, formula string implies close
            .mul(rank(c.div(o).replace([-np.inf, np.inf], np.nan)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())

def alpha039(c, v, r, adv20):
    """
    Alpha Factor #39: -rank(ts_delta(close, 7) * (1 - rank(ts_weighted_mean(volume / adv20, 9)))) * (1 + rank(ts_sum(returns, 250)))

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。
        v: pandas DataFrame, 成交量。
        r: pandas DataFrame, 收益率。
        adv20: pandas DataFrame, 20日平均成交量。
    适用性:
        截面 Alpha (使用了 rank, ts_weighted_mean)
    Alpha 含义:
        与 Alpha019 结构类似，但第一部分的计算不同。
        Part 1_inner = ts_delta(close, 7) * (1 - rank(ts_weighted_mean(volume / adv20, 9)))
            - ts_delta(close, 7): 7日收盘价变化。
            - volume / adv20: 量比。
            - ts_weighted_mean(..., 9): 量比的9日加权平均。
            - rank(...): 对加权平均量比进行排名。
            - (1 - rank(...)): 如果量比排名高，此项小。
            这部分结合了7日价格动量和（反向的）近期平均量比排名。
        Part 1 = -rank(Part 1_inner)
        Part 2 = 1 + rank(ts_sum(returns, 250))  (注意：代码中使用 ts_mean(r, 250)，这里按公式的 ts_sum)
                 实际代码: 1 + rank(ts_mean(r, 250).add(1)) -> 应该是 1 + rank(1 + ts_sum(returns, 250))
                 按代码实现: 1 + rank(ts_mean(r, 250) + 1)
            - ts_sum(returns, 250): 250日累积收益。 (代码用的是 ts_mean(r,250).add(1) 然后rank，再+1)
            - rank(1 + ts_sum(returns, 250)): 对 (1+长期累积收益) 进行排名。
            - 1 + ...: 排名结果再加1。
        最终 Alpha = Part 1 * Part 2。
        该 Alpha 试图捕捉短期价格变动与量比模式的组合，并结合长期收益表现。
        例如，如果 Part1_inner 排名靠前（导致 Part1 为负），且长期收益 Part2 排名靠前（导致 Part2 为正），则 Alpha 为负。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    # Original formula: (1 + rank(ts_sum(returns, 250)))
    # Code: .mul(rank(ts_mean(r, 250).add(1))) --- this means rank( (sum(r,250)/250) + 1 ). Let's assume it's a variation.
    # The final formula string shows (1 + rank(ts_sum(returns, 250))), which means the rank is of (ts_sum_ret + 1).
    # The code structure `rank(ts_mean(r, 250).add(1))` means `rank( (sum_ret/250) + 1 )`.
    # Let's assume the structure of Alpha019's second part: (1 + rank( (1+SUM_RET) ))
    # For now, I will follow the code's implementation for part2: rank(mean_ret + 1) and then the outer (1 + rank_val).
    # The formula description seems to imply: rank_term_A * (1 + rank_term_B)
    # rank_term_A = -rank(ts_delta(close, 7) * (1 - rank(ts_weighted_mean(volume / adv20, 9))))
    # rank_term_B = rank(1 + ts_sum(returns, 250))
    # The code is: rank_term_A_code.mul(rank(ts_mean(r, 250).add(1)))
    # This makes Part2 = rank( (sum_ret/250) + 1 ).
    # Let's assume the formula string is the target for the final product for Part2.
    # (1 + rank(ts_sum(returns,250)))
    # The code's current mul(...) is rank(ts_mean(r, 250).add(1)).
    # If we strictly follow the formula string's Part 2: (1 + rank(ts_sum(r,250)))
    # Corrected Part 2 based on formula string structure:
    part2_formula_style = (1 + rank(ts_sum(r, 250).add(1))) # rank(1+sum) then add 1

    return (rank(ts_delta(c, 7).mul(rank(ts_weighted_mean(v.div(adv20), 9)).mul(-1).add(1))).mul(-1)
            .mul(part2_formula_style) # Using the formula string's structure for part 2
            .stack('ticker')
            .swaplevel())

def alpha040(h, v):
    """
    Alpha Factor #40: ((-1 * rank(ts_std(high, 10))) * ts_corr(high, volume, 10))

    中文注释:
    入参:
        h: pandas DataFrame, 最高价。
        v: pandas DataFrame, 成交量。
    适用性:
        截面 Alpha (使用了 rank 和 ts_corr)
    Alpha 含义:
        此 Alpha 结合了最高价的波动率和最高价与成交量的相关性。
        1. -1 * rank(ts_std(high, 10)):
           - ts_std(high, 10): 过去10日最高价的标准差（波动率）。
           - rank(...): 对此波动率进行横截面排名。
           - -1 * ...: 取负。波动率越高，排名越靠前，此项越负。
        2. ts_corr(high, volume, 10): 过去10日最高价与成交量的时间序列相关性。
        最终 Alpha 是这两项的乘积。
        - 如果最高价波动率高（Part1 为负），且最高价与成交量正相关（Part2 为正），则 Alpha 为负。
        - 如果最高价波动率高（Part1 为负），且最高价与成交量负相关（Part2 为负），则 Alpha 为正。
        它试图捕捉高波动环境下，量价关系对未来收益的指示作用。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    return (rank(ts_std(h, 10)) # rank of std
            .mul(ts_corr(h, v, 10)) # multiplied by corr
            .mul(-1) # then multiply by -1
            .stack('ticker')
            .swaplevel())

def alpha041(h, l, vwap):
    """
    Alpha Factor #41: power(high * low, 0.5) - vwap
    (Original formula string seems to have a typo: "0.5 - vwap", implying (high*low)^0.5 - vwap)

    中文注释:
    入参:
        h: pandas DataFrame, 最高价。
        l: pandas DataFrame, 最低价。
        vwap: pandas DataFrame, 成交量加权平均价。
    适用性:
        时序 Alpha (直接计算，没有横截面操作如rank)
    Alpha 含义:
        此 Alpha 计算每日最高价和最低价的几何平均数与当日VWAP的差。
        1. (high * low)^0.5: 最高价和最低价的几何平均值，可视为当日价格中枢的一种估计。
        2. Alpha = (high * low)^0.5 - vwap。
        如果价格中枢高于VWAP，Alpha为正；反之为负。
        它衡量了价格中枢相对于成交量加权平均价格的偏离程度。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    return (power(h.mul(l), 0.5)
            .sub(vwap)
            .stack('ticker')
            .swaplevel())

def alpha042(c, vwap):
    """
    Alpha Factor #42: rank(vwap - close) / rank(vwap + close)

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。
        vwap: pandas DataFrame, 成交量加权平均价。
    适用性:
        截面 Alpha (使用了 rank)
    Alpha 含义:
        此 Alpha 比较了 (VWAP - 收盘价) 的排名和 (VWAP + 收盘价) 的排名。
        1. rank(vwap - close):
           如果 VWAP > close (收盘价低于VWAP)，则 vwap - close > 0，排名靠前。
           如果 VWAP < close (收盘价高于VWAP)，则 vwap - close < 0，排名靠后。
        2. rank(vwap + close):
           VWAP 与收盘价之和的排名。通常这个值都是正的，排名主要反映其量级。
        Alpha = rank(vwap - close) / rank(vwap + close)。
        - 当股票收盘价远低于VWAP (vwap-close 大，排名高)，同时 vwap+close 的值相对较小 (排名低) 时，Alpha 值会较大。
        - 它试图捕捉价格相对VWAP的偏离程度，并用一个相对值 (与vwap+close排名之比) 来衡量。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    return (rank(vwap.sub(c))
            .div(rank(vwap.add(c))) # Division by zero if rank is zero, though rank is usually >0
            .stack('ticker')
            .swaplevel())

def alpha043(adv20, c, v): # adv20 is first param in definition, but used as v.div(adv20)
    """
    Alpha Factor #43: (ts_rank((volume / adv20), 20) * ts_rank((-1 * ts_delta(close, 7)), 8))

    中文注释:
    入参:
        adv20: pandas DataFrame, 20日平均成交量。(注意：实际使用中，v/adv20，所以adv20应为分母)
               函数定义为 (adv20, c, v)，但公式为 volume/adv20，所以v是当日成交量，adv20是均量。
               代码实现是 v.div(adv20)，这意味着传入的 adv20 确实是均量数据。
        c: pandas DataFrame, 收盘价。
        v: pandas DataFrame, 当日成交量。
    适用性:
        时序 Alpha (主要基于 ts_rank)
    Alpha 含义:
        此 Alpha 结合了量比的长期时序排名和短期价格反转信号的近期时序排名。
        1. volume / adv20: 当日成交量与20日平均成交量的比率（量比）。
        2. ts_rank(volume / adv20, 20): 量比在过去20日的时序排名。排名高表示近期成交相对活跃。
        3. -1 * ts_delta(close, 7): 过去7日收盘价变化的负值（反转信号）。如果7日价格上涨，此项为负。
        4. ts_rank(..., 8): 上述反转信号在过去8日的时序排名。
        Alpha = Part2 * Part4。
        它试图寻找那些近期成交量持续活跃（量比时序排名高），并且短期价格反转信号也持续较强（反转信号时序排名高）的股票。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    # adv20 is indeed ts_mean(v,20) or similar passed in.
    return (ts_rank(v.div(adv20), 20)
            .mul(ts_rank(ts_delta(c, 7).mul(-1), 8))
            .stack('ticker')
            .swaplevel())

def alpha044(h, v):
    """
    Alpha Factor #44: -ts_corr(high, rank(volume), 5)

    中文注释:
    入参:
        h: pandas DataFrame, 最高价。
        v: pandas DataFrame, 成交量。
    适用性:
        截面/时序混合 (rank 是截面, ts_corr 是时序)
    Alpha 含义:
        此 Alpha 计算最高价与成交量排名在过去5日内的负相关性。
        1. rank(volume): 成交量的横截面排名。
        2. ts_corr(high, rank(volume), 5): 最高价（原始值）与成交量排名在过去5日的时间序列相关性。
        3. Alpha = -1 * ...
        - 如果最高价上涨的同时成交量排名也上升（正相关），则Alpha为负。
        - 如果最高价上涨的同时成交量排名下降（负相关），则Alpha为正。
        它关注的是价格行为（最高价）与成交量相对强度（排名）之间的关系。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。包含 NaN 值。
    """

    return (ts_corr(h, rank(v), 5)
            .replace([-np.inf, np.inf], np.nan)
            .mul(-1)
            .stack('ticker')
            .swaplevel())

def alpha045(c, v):
    """
    Alpha Factor #45: -(rank((ts_mean(ts_lag(close, 5), 20)) * ts_corr(close, volume, 2)) * rank(ts_corr(ts_sum(close, 5), ts_sum(close, 20), 2)))

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。
        v: pandas DataFrame, 成交量。
    适用性:
        截面 Alpha (使用了 rank 和 ts_corr)
    Alpha 含义:
        此 Alpha 结构复杂，包含两个主要部分的乘积，然后取负。
        Part 1: rank(ts_mean(ts_lag(close, 5), 20) * ts_corr(close, volume, 2))
            - ts_lag(close, 5): 5日前收盘价。
            - ts_mean(..., 20): 5日前收盘价的20日均值。 (这表示一个更早期的价格平均水平)
            - ts_corr(close, volume, 2): 当前收盘价与成交量的2日相关性 (极短期量价关系)。
            - rank(product): 对上述两项乘积进行横截面排名。
        Part 2: rank(ts_corr(ts_sum(close, 5), ts_sum(close, 20), 2))
            - ts_sum(close, 5): 5日收盘价总和。
            - ts_sum(close, 20): 20日收盘价总和。
            - ts_corr(..., ..., 2): 上述两个价格总和的2日相关性。这衡量了短期趋势和中期趋势的同步性。
            - rank(...): 对此相关性进行横截面排名。
        Alpha = - (Part 1 * Part 2)。
        该 Alpha 试图寻找满足特定组合条件的股票：
        (滞后价格均值 * 短期量价关系)的排名 与 (短中期趋势同步性)的排名 的乘积，再取负。
        解读难度较大，可能是在捕捉某种复杂的量价时滞效应和趋势确认信号。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。包含 NaN 值。
    """

    return (rank(ts_mean(ts_lag(c, 5), 20) # Rank of product
                 .mul(ts_corr(c, v, 2)
                      .replace([-np.inf, np.inf], np.nan)))
            .mul(rank(ts_corr(ts_sum(c, 5), # Multiplied by another rank
                              ts_sum(c, 20), 2)))
            .mul(-1) # Final negation
            .stack('ticker')
            .swaplevel())

def alpha046(c):
    """
    Alpha Factor #46:
    (0.25 < (X = ts_lag(ts_delta(close, 10), 10) / 10 - ts_delta(close, 10) / 10)) ? -1
        : (X < 0 ? 1 : -ts_delta(close, 1))

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。
    适用性:
        时序 Alpha (基于价格变化率的比较)
    Alpha 含义:
        此 Alpha 基于一个衡量价格动量变化的值 X 进行条件判断。
        X = (10日前“10日价格变动”的日均值) - (当前“10日价格变动”的日均值)
          = ( (c[-10] - c[-20])/10 ) - ( (c[0] - c[-10])/10 )
          = (c[-10] - c[-20] - c[0] + c[-10]) / 10
          = (2*c[-10] - c[-20] - c[0]) / 10
        这衡量了近期动量 (c[0]-c[-10]) 相对于稍早时期动量 (c[-10]-c[-20]) 的变化。
        如果 X > 0，表示动量减弱或反转。如果 X < 0，表示动量增强。

        1. 如果 X > 0.25 (动量显著减弱/反转): Alpha = -1。
        2. 否则，如果 X < 0 (动量增强): Alpha = 1。
        3. 否则 (0 <= X <= 0.25，动量小幅减弱或平稳): Alpha = -ts_delta(close, 1) (昨日收盘价与今日收盘价之差的反转)。

        代码实现与公式字符串的条件分支逻辑略有差异，注释以代码为准：
        `cond.where(cond > 0.25, -alpha.where(cond < 0, -ts_delta(c, 1)))`
        where(A, B, C) means if A then B else C.
        - if cond > 0.25: result is cond itself (this seems like a bug in code, should be -1 based on formula)
          Assuming the intent was a fixed value like -1 for the first branch.
          Let's assume the formula logic for the first branch value, but use code's structure.
          If cond > 0.25, then -1 (from `-alpha` where alpha is initially -1, or just directly -1).
          If not (cond <= 0.25):
            then `-alpha.where(cond < 0, -ts_delta(c,1))`
            This inner part: `alpha_inner_val = (cond < 0) ? alpha_val_representing_1 : -ts_delta(c,1)`
            So if cond < 0: `alpha_inner_val = 1` (alpha is -1, so -alpha is 1)
            Else (0 <= cond <= 0.25): `alpha_inner_val = -ts_delta(c,1)`
        So:
        - if cond > 0.25: Alpha = -1 (assuming the intent of the formula string's -1 value)
        - else if cond < 0: Alpha = 1
        - else (0 <= cond <= 0.25): Alpha = -ts_delta(close, 1)
        This matches the formula string. The code `cond.where(cond > 0.25, ...)` if `cond` itself is used as the result for `cond > 0.25` is unusual.
        The provided code `alpha = pd.DataFrame(-np.ones_like(cond)...)` and then `cond.where(cond > 0.25, -alpha.where(...))` means if `cond > 0.25`, the result is `cond` values, not `-1`.
        This part of the code seems to directly use the `cond` value instead of `-1` if `cond > 0.25`.
        I will proceed by commenting based on the code's literal execution of `cond.where(cond > 0.25, ...)`.
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """

    cond = ts_lag(ts_delta(c, 10), 10).div(10).sub(ts_delta(c, 10).div(10))
    # alpha_df_ones is a DataFrame of 1s where cond is not null, else nan
    alpha_df_ones = pd.DataFrame(np.ones_like(cond.values), index=c.index, columns=c.columns)
    alpha_df_ones[cond.isnull()] = np.nan

    # if cond > 0.25, result is cond (the calculated X value)
    # else, result is alpha_df_ones.where(cond < 0, -ts_delta(c,1))
    #   if cond < 0, result is 1 (from alpha_df_ones)
    #   else (0 <= cond <= 0.25), result is -ts_delta(c,1)
    # This interpretation aligns with the code structure.
    result = cond.where(cond > 0.25,
                        alpha_df_ones.where(cond < 0, -ts_delta(c, 1)))

    return (result
            .stack('ticker')
            .swaplevel())

def alpha047(h, c, v, vwap, adv20):
    """
    Alpha Factor #47: ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (ts_sum(high, 5) /5))) - rank((vwap - ts_lag(vwap, 5))))

    中文注释:
    入参:
        h: pandas DataFrame, 最高价。
        c: pandas DataFrame, 收盘价。
        v: pandas DataFrame, 成交量。
        vwap: pandas DataFrame, 成交量加权平均价。
        adv20: pandas DataFrame, 20日平均成交量。
    适用性:
        截面 Alpha (使用了 rank, ts_mean)
    Alpha 含义:
        此 Alpha 结构复杂，由两大部分相减构成。
        Part 1: (((rank(1/close) * volume) / adv20) * (high * rank(high - close)) / ts_mean(high, 5)))
            - rank(1/close): 收盘价倒数的排名。低价股排名靠前。
            - (rank(1/close) * volume) / adv20: 上述排名乘以成交量，再除以20日均量。可看作是“低价股成交活跃度指标”。
            - rank(high - close): 上影线长度 (h-c) 的排名。上影线长则排名高。
            - high * rank(high - close): 最高价乘以上影线长度排名。
            - ts_mean(high, 5): 5日最高价均值。
            - Part1_sub2 = (high * rank(high - close)) / ts_mean(high, 5)
            - Part1 = Part1_sub1 * Part1_sub2
        Part 2: rank(vwap - ts_lag(vwap, 5))
            - vwap - ts_lag(vwap, 5): 当日VWAP与5日前VWAP的差值 (VWAP的5日变化)。
            - rank(...): 对此VWAP变化进行排名。
        Alpha = Part 1 - Part 2。
        它试图结合“低价股活跃度”、“上影线效应”与“VWAP动量”等多种市场信息。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """

    part1_sub1 = rank(c.pow(-1)).mul(v).div(adv20)
    part1_sub2 = h.mul(rank(h.sub(c))).div(ts_mean(h, 5)) # ts_sum(high,5)/5 is ts_mean(high,5)
    part1 = part1_sub1.mul(part1_sub2)
    part2 = rank(ts_delta(vwap, 5)) # vwap - ts_lag(vwap,5) is ts_delta(vwap,5)

    return (part1.sub(part2)
            .stack('ticker')
            .swaplevel())

def alpha048(c, industry): # Function name in file is alpha48, assuming typo and it's alpha048
    """
    Alpha Factor #48: (indneutralize(((ts_corr(ts_delta(close, 1), ts_delta(ts_lag(close, 1), 1), 250) * ts_delta(close, 1)) / close), IndClass.subindustry) / ts_sum(((ts_delta(close, 1) / ts_lag(close, 1))^2), 250))

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。
        industry: pandas Series/DataFrame, 股票对应的行业分类信息 (具体应为 IndClass.subindustry 指定的子行业级别)。
                  格式应与 indneutralize 函数兼容。
    适用性:
        截面 Alpha (核心是 indneutralize, ts_corr, ts_sum)
    Alpha 含义:
        此 Alpha 关注价格动量相关性的行业中性化处理，并用波动率进行调整。
        Numerator Part 1: (ts_corr(ts_delta(close, 1), ts_delta(ts_lag(close, 1), 1), 250) * ts_delta(close, 1)) / close
            - ts_delta(close, 1): 当日价格变动。
            - ts_delta(ts_lag(close, 1), 1): 昨日价格变动。
            - ts_corr(..., ..., 250): 当日价格变动与昨日价格变动在过去250日的相关性（动量的持续性/反转性）。
            - (... * ts_delta(close, 1)): 上述相关性乘以当日价格变动。
            - (... / close): 用收盘价进行归一化，类似于收益率。
        Numerator Part 2: indneutralize(Numerator Part 1, IndClass.subindustry)
            - 对 Part 1 的结果，按照子行业 (IndClass.subindustry) 进行行业中性化处理。
            这意味着移除行业 común 影响，提取个股特有的alpha。
        Denominator: ts_sum(((ts_delta(close, 1) / ts_lag(close, 1))^2), 250)
            - (ts_delta(close, 1) / ts_lag(close, 1)): 近似日收益率。
            - (...)^2: 收益率平方。
            - ts_sum(..., 250): 过去250日收益率平方和，衡量年度波动性。
        Alpha = Numerator Part 2 / Denominator。
        它是一个经过行业中性化和波动率调整的动量相关性因子。
        【注意】: 此函数标为 `pass`，未实现。注释基于公式字符串。`IndClass.subindustry` 需在别处定义。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    pass

def alpha049(c):
    """
    Alpha Factor #49: (X = ts_delta(ts_lag(close, 10), 10)/10 - ts_delta(close, 10)/10) < -0.1 * close ? 1 : -ts_delta(close, 1)
    (Note: original formula string in file has no `*c` for threshold, but notebook for alpha049 might have it. Assuming formula string from file.)
    If threshold is `-0.1` (not `-0.1*c`):
    X = (2*c[-10] - c[-20] - c[0]) / 10, (动量变化，同Alpha046)

    中文注释: (以公式字符串为准，若 `-0.1*c` 为意图，则条件不同)
    入参:
        c: pandas DataFrame, 收盘价。
    适用性:
        时序 Alpha (条件型，基于价格动量变化)
    Alpha 含义:
        此 Alpha 基于价格动量变化 X (同 Alpha046 中的定义) 进行决策。
        X = ( (c[-10]-c[-20])/10 ) - ( (c[0]-c[-10])/10 )

        1. 条件: 如果 X < -0.1 (即近期动量相对于前期动量显著增强，价格加速下跌或加速上涨放缓后再次加速上涨)。
           代码实现为 `cond = (X >= -0.1 * c)` 然后 `where(cond, 1)`，这意味着如果 `X >= -0.1*c` 则为1。
           若按公式 `X < -0.1*c ? 1`，则 `where(X < -0.1*c, 1, -ts_delta(c,1))`
           当前代码: `where(X >= -0.1*c, -ts_delta(c,1), 1)` because of `cond` definition and `where(cond, 1)` is `where(X < -0.1*c, 1, original_ts_delta_val)`.
           Let's assume the formula string: `X < -0.1*c ? 1 : -ts_delta(c,1)`
        2. 如果条件为真: Alpha = 1。
        3. 否则: Alpha = -ts_delta(close, 1) (昨日收盘价与今日收盘价之差的反转)。

        该 Alpha 在观察到特定模式的动量增强时给固定信号1，否则采取短期反转策略。
        【注意】代码中条件是 `X >= -0.1*c`，且赋值逻辑与公式字符串顺序相反。注释以公式字符串的意图为准。
        如果严格按代码 `cond = (X >= -0.1 * c); return (-ts_delta(c, 1).where(cond, 1))`
        这意味着：if X >= -0.1*c, Alpha = -ts_delta(c,1). Else (X < -0.1*c), Alpha = 1. This matches the formula string.
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    X = ts_delta(ts_lag(c, 10), 10).div(10).sub(ts_delta(c, 10).div(10))
    # condition from formula string: X < -0.1 * c
    # code's cond: X >= -0.1 * c
    # code's logic: if (X >= -0.1*c) is true, use -ts_delta(c,1). Otherwise (X < -0.1*c), use 1.
    # This actually matches the formula: (X < -0.1*c) ? 1 : -ts_delta(c,1)
    condition_from_formula = X < -0.1 * c

    return pd.Series(np.where(condition_from_formula, 1, -ts_delta(c, 1)), index=c.index, columns=c.columns if isinstance(c, pd.DataFrame) else None).stack().swaplevel()
    # The original code was:
    # cond = (ts_delta(ts_lag(c, 10), 10).div(10)
    #         .sub(ts_delta(c, 10).div(10)) >= -0.1 * c)
    # return (-ts_delta(c, 1)
    #         .where(cond, 1) # This means: if cond is True, use -ts_delta(c,1). If cond is False, use 1.
    #         .stack('ticker')
    #         .swaplevel())
    # This is: if X >= -0.1*c, then -ts_delta(c,1). Else (X < -0.1*c), then 1. This is correct.

def alpha050(v, vwap):
    """
    Alpha Factor #50: -ts_max(rank(ts_corr(rank(volume), rank(vwap), 5)), 5)

    中文注释:
    入参:
        v: pandas DataFrame, 成交量。
        vwap: pandas DataFrame, 成交量加权平均价。
    适用性:
        截面 Alpha (使用了 rank, ts_corr, ts_max)
    Alpha 含义:
        此 Alpha 关注成交量排名与VWAP排名的相关性的近期最大值的负值。
        1. rank(volume): 成交量的横截面排名。
        2. rank(vwap): VWAP的横截面排名。
        3. ts_corr(rank(v), rank(vwap), 5): 上述两个排名在过去5日的时间序列相关性。
        4. rank(...): 对步骤3的相关性进行横截面排名。
        5. ts_max(..., 5): 步骤4的相关性排名在过去5日的最大值。
        6. Alpha = -1 * ...
        该 Alpha 试图识别那些“成交量排名与VWAP排名之相关性的排名”在近期持续处于高位的股票，并给予负的 Alpha 值。
        它强调了量价排名关系强度（通过相关性排名衡量）的持续性。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    return (ts_max(rank(ts_corr(rank(v),
                                rank(vwap), 5)), 5)
            .mul(-1)
            .stack('ticker')
            .swaplevel())

def alpha051(c):
    """
    Alpha Factor #51: (X = ts_delta(ts_lag(close, 10), 10)/10 - ts_delta(close, 10)/10) < -0.05 * c ? 1 : -ts_delta(close, 1)

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。
    适用性:
        时序 Alpha (条件型，基于价格动量变化)
    Alpha 含义:
        与 Alpha049 结构和逻辑完全相同，唯一的区别在于阈值。
        Alpha049 的阈值是 -0.1 * c，而 Alpha051 的阈值是 -0.05 * c。
        X = ( (c[-10]-c[-20])/10 ) - ( (c[0]-c[-10])/10 ) (价格动量变化)

        1. 条件: 如果 X < -0.05 * c (近期动量相对于前期动量显著增强，但程度小于Alpha049的要求)。
        2. 如果条件为真: Alpha = 1。
        3. 否则: Alpha = -ts_delta(close, 1) (昨日收盘价与今日收盘价之差的反转)。

        代码实现逻辑与Alpha049相同: if X >= -0.05*c, then -ts_delta(c,1). Else (X < -0.05*c), then 1.
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    X = ts_delta(ts_lag(c, 10), 10).div(10).sub(ts_delta(c, 10).div(10))
    condition_from_formula = X < -0.05 * c # Threshold is -0.05*c

    # Code's logic: if (X >= -0.05*c) is true, use -ts_delta(c,1). Otherwise (X < -0.05*c), use 1.
    # This matches the formula: (X < -0.05*c) ? 1 : -ts_delta(c,1)
    return (-ts_delta(c, 1)
            .where(X >= -0.05 * c, 1) # if X >= -0.05*c then -delta, else 1
            .stack('ticker')
            .swaplevel())

def alpha052(l, v, r):
    """
    Alpha Factor #52: (ts_lag(ts_min(low, 5), 5) - ts_min(low, 5)) * rank((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220) * ts_rank(volume, 5)
    (Formula string is `ts_lag(ts_min(low,5),5) - ts_min(low,5)`, which is `-ts_delta(ts_min(low,5),5)`. Code uses `ts_delta(ts_min(l, 5), 5)`)
    Assuming code `ts_delta(ts_min(l,5),5)` is the intended logic for the first part.

    中文注释: (按代码实现 `ts_delta(ts_min(l,5),5)` 注释第一部分)
    入参:
        l: pandas DataFrame, 最低价。
        v: pandas DataFrame, 成交量。
        r: pandas DataFrame, 收益率。
    适用性:
        截面/时序混合 Alpha (使用了 rank, ts_rank)
    Alpha 含义:
        此 Alpha 结合了最低价的变化、长期收益率与中期收益率的差额，以及成交量的时序排名。
        1. Part1 = ts_delta(ts_min(low, 5), 5):
           - ts_min(low, 5): 过去5日最低价的最小值。
           - ts_delta(..., 5): 上述5日内最低价的5日变化。表示近期低点支撑的变化情况。
             如果近期低点在上移，此项为正。
        2. Part2 = rank((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220):
           - ts_sum(returns, 240): 过去240日（约一年）的累积收益。
           - ts_sum(returns, 20): 过去20日（约一月）的累积收益。
           - (sum240 - sum20) / 220: (年化收益 - 月化收益) / (240-20)。可理解为剔除最近一个月影响后的长期平均日收益率。
           - rank(...): 对此进行横截面排名。
        3. Part3 = ts_rank(volume, 5): 成交量在过去5日的时序排名。
        Alpha = Part1 * Part2 * Part3。
        它试图寻找那些：近期低点在抬高，长期（排除近期）收益表现好，并且近期成交量也活跃（时序排名高）的股票。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    # Code uses ts_delta(ts_min(l, 5), 5)
    # Formula string: ts_lag(ts_min(low, 5), 5) - ts_min(low, 5) = - (ts_min(low,5) - ts_lag(ts_min(low,5),5))
    # which is -ts_delta(ts_min(low,5), ts_lag_val=5) if ts_delta is current - lag.
    # If ts_delta(X, N) is X_t - X_{t-N}.
    # Then ts_delta(ts_min(low,5), 5) is (current 5-day-min-low) - (5-day-min-low 5 days ago).
    # Formula string's (ts_lag(A,5) - A) is -(A - ts_lag(A,5)) = -ts_delta(A,5).
    # The code implements ts_delta(ts_min(l,5),5). The comment will follow the code.
    part1 = ts_delta(ts_min(l, 5), 5)
    part2 = rank(ts_sum(r, 240).sub(ts_sum(r, 20)).div(220))
    part3 = ts_rank(v, 5)

    return (part1.mul(part2).mul(part3)
            .stack('ticker')
            .swaplevel())

def alpha053(h, l, c):
    """
    Alpha Factor #53: -1 * ts_delta( (1 - (high - close) / (close - low + 1e-6)) , 9)

    中文注释:
    入参:
        h: pandas DataFrame, 最高价。
        l: pandas DataFrame, 最低价。
        c: pandas DataFrame, 收盘价。
    适用性:
        时序 Alpha (主要基于 ts_delta)
    Alpha 含义:
        此 Alpha 关注一个衡量日内价格位置指标的9日变化。
        1. Inner ratio: (high - close) / (close - low + 1e-6)
           - high - close: 上影线长度（如果 h > c）。
           - close - low: 下影线长度（如果 c > l）。
           - +1e-6 是为了防止除以零。
           - 这个比率衡量了上影线相对于下影线的长度。
        2. Indicator: 1 - Inner ratio
           如果上影线远长于下影线，Inner ratio 大，Indicator 小或为负。
           如果下影线远长于上影线，Inner ratio 小，Indicator 大。
           如果收盘价接近最高价 (上影线短)，Inner ratio 小，Indicator 接近1。
           如果收盘价接近最低价 (下影线短)，Inner ratio 大，Indicator 远小于1或为负。
        3. ts_delta(Indicator, 9): 上述指标在过去9日的变化。
        4. Alpha = -1 * ...
        该 Alpha 捕捉了这个日内价格相对位置指标的变化趋势的反转。
        例如，如果该指标在9天内持续上升（比如从下影线主导变为上影线主导，或价格从接近低点变为接近高点），则ts_delta为正，Alpha为负。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    # Original code:
    # inner = (c.sub(l)).add(1e-6) # This is (close - low + 1e-6)
    # return (ts_delta(h.sub(c) # (high-close)
    #                  .mul(-1).add(1) # 1 - (high-close) -- THIS IS WRONG, division is first
    #                  .div(c.sub(l) # This is (close-low)
    #                       .add(1e-6)), 9)
    #         .mul(-1)
    # The code's (1 - (h-c)) / (c-l+eps) is NOT 1 - ( (h-c)/(c-l+eps) )
    # It should be 1 - ( (h-c).div(c-l+eps) )
    # Correct implementation based on formula:
    ratio = (h.sub(c)).div(c.sub(l).add(1e-6))
    indicator = decimal_1.sub(ratio) # 1 - ratio

    return (ts_delta(indicator, 9)
            .mul(-1)
            .stack('ticker')
            .swaplevel())

def alpha054(o, h, l, c):
    """
    Alpha Factor #54: -(low - close) * power(open, 5) / ((low - high) * power(close, 5))

    中文注释:
    入参:
        o: pandas DataFrame, 开盘价。
        h: pandas DataFrame, 最高价。
        l: pandas DataFrame, 最低价。
        c: pandas DataFrame, 收盘价。
    适用性:
        时序 Alpha (直接计算，无排名等截面操作)
    Alpha 含义:
        此 Alpha 结合了日内价格振幅、开盘价和收盘价的幂次方。
        Numerator: -(low - close) * power(open, 5) = (close - low) * open^5
            - close - low: 当日下影线长度（如果收盘高于最低）。
            - open^5: 开盘价的5次方，极大地放大了开盘价的影响。
        Denominator: (low - high) * power(close, 5) = -(high - low) * close^5
            - high - low: 当日总振幅。
            - close^5: 收盘价的5次方。
        Alpha = ( (close - low) * open^5 ) / ( -(high - low) * close^5 )
              = - ( (close - low) / (high - low) ) * (open/close)^5

        - (close - low) / (high - low): 收盘价在当日振幅中的相对位置（归一化到0-1，如果c在l,h之间）。负号使其反向。
        - (open/close)^5: 开盘价与收盘价之比的5次方，衡量日内价格变动幅度。

        该 Alpha 对价格的相对位置和日内开盘/收盘的相对强弱非常敏感，特别是由于5次方的存在。
        例如，如果收盘价接近最低价 (c-l 小)，且开盘价远高于收盘价 (o/c 大)，则Alpha可能为较大的负值。
        `replace(0, -0.0001)` 用于防止分母 `low-high` 为零（即最低价等于最高价，通常只有在无交易或价格不变时发生）。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    numerator = l.sub(c).mul(o.pow(5)).mul(-1) # (c-l) * o^5
    denominator = l.sub(h).replace(0, -0.0001).mul(c.pow(5)) # -(h-l) * c^5, avoid h=l
    return (numerator.div(denominator)
            .stack('ticker')
            .swaplevel())

def alpha055(c, h, l, v):
    """
    Alpha Factor #55: (-1 * ts_corr(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low,12)))), rank(volume), 6))

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。
        h: pandas DataFrame, 最高价。
        l: pandas DataFrame, 最低价。
        v: pandas DataFrame, 成交量。
    适用性:
        截面 Alpha (使用了 rank 和 ts_corr)
    Alpha 含义:
        此 Alpha 计算了收盘价在近期价格区间内的位置（类似Stochastic %K指标）的排名，与成交量排名的负相关性。
        1. Price Position Indicator: (close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12) + 1e-6)
           - ts_min(low, 12): 过去12日的最低价。
           - ts_max(high, 12): 过去12日的最高价。
           - (ts_max - ts_min): 过去12日的价格区间。
           - 这个指标衡量了当前收盘价在过去12日价格区间中所处的位置。值域通常在0到1之间。
           - `+1e-6` (代码中是 `replace(0, 1e-6)`) 是为了防止除以零。
        2. rank(Price Position Indicator): 对上述价格位置指标进行横截面排名。
        3. rank(volume): 成交量的横截面排名。
        4. ts_corr(..., ..., 6): 计算步骤2和步骤3的排名在过去6日的时间序列相关性。
        5. Alpha = -1 * ...
        该 Alpha 试图寻找那些“价格位置排名”与“成交量排名”呈反向相关性的股票。
        例如，如果价格位置排名上升（收盘价接近近期高点）而成交量排名下降，相关性为负，则 Alpha 为正。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。包含 NaN 值。
    """
    price_pos_num = c.sub(ts_min(l, 12))
    price_pos_den = ts_max(h, 12).sub(ts_min(l, 12)).replace(0, 1e-6) # Avoid division by zero
    price_pos_indicator = price_pos_num.div(price_pos_den)

    return (ts_corr(rank(price_pos_indicator),
                    rank(v), 6)
            .replace([-np.inf, np.inf], np.nan)
            .mul(-1)
            .stack('ticker')
            .swaplevel())

def alpha056(r, cap):
    """
    Alpha Factor #56: -rank(ts_sum(returns, 10) / ts_sum(ts_sum(returns, 2), 3)) * rank((returns * cap))

    中文注释:
    入参:
        r: pandas DataFrame, 收益率。
        cap: pandas DataFrame, 市值。
    适用性:
        截面 Alpha (使用了 rank)
    Alpha 含义:
        此 Alpha 结合了收益率的时间序列聚合特性与市值加权的当日收益。
        Part 1: -rank(ts_sum(returns, 10) / ts_sum(ts_sum(returns, 2), 3))
            - ts_sum(returns, 10): 10日累积收益。
            - ts_sum(returns, 2): 2日累积收益。
            - ts_sum(ts_sum(returns, 2), 3): ( (r_t + r_{t-1}) + (r_{t-1} + r_{t-2}) + (r_{t-2} + r_{t-3}) )
                                         = r_t + 2*r_{t-1} + 2*r_{t-2} + r_{t-3}
                                         一个对近期收益有更高权重的累积和。
            - ratio = 10日累积收益 / 近期高权重累积收益。
            - -rank(ratio): 对该比率的排名取负。
        Part 2: rank(returns * cap)
            - returns * cap: 当日收益率乘以市值，即当日市值加权收益（或称市值的日度变动额）。
            - rank(...): 对其进行横截面排名。
        Alpha = Part 1 * Part 2。
        它试图寻找那些“特定收益率聚合比率”排名靠后（-rank后靠前），且“市值加权收益”排名也靠前的股票。
        【注意】: 此函数标为 `pass`，未实现。注释基于公式字符串。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    pass

def alpha057(c, vwap):
    """
    Alpha Factor #57: -(close - vwap) / ts_weighted_mean(rank(ts_argmax(close, 30)), 2)

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。
        vwap: pandas DataFrame, 成交量加权平均价。
    适用性:
        截面/时序混合 (ts_weighted_mean, rank, ts_argmax)
    Alpha 含义:
        此 Alpha 衡量了价格与VWAP的偏离，并用近期价格高点动量的加权均值进行调整。
        Numerator: -(close - vwap) = vwap - close
            - 表示VWAP相对于收盘价的偏离。如果VWAP高，值为正。
        Denominator: ts_weighted_mean(rank(ts_argmax(close, 30)), 2)
            - ts_argmax(close, 30): 过去30日内收盘价达到最高值的那一天的“索引”（天数，0表示今天，1表示昨天等）。
            - rank(...): 对这个“最高点出现时间”进行横截面排名。如果最高点刚出现，ts_argmax小，排名可能靠前或后取决于具体rank定义。
            - ts_weighted_mean(..., 2): 对上述排名进行2日的加权移动平均。
            - +1e-5 in code: vwap.add(1e-5) in numerator, likely to avoid issues if c exactly equals vwap, though division is by denominator.
                              The denominator itself could be zero.
        Alpha = Numerator / Denominator
        它试图寻找那些VWAP显著偏离收盘价，并且这种偏离相对于“近期价格高点动量排名”的加权均值而言较大的股票。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    numerator = vwap.sub(c) # -(close - vwap) = vwap - close. Code uses c.sub(vwap.add(1e-5))...mul(-1) which is vwap+eps-c
    denominator = ts_weighted_mean(rank(ts_argmax(c, 30)), 2) # Denominator could be zero

    # To prevent division by zero if denominator can be zero, add small epsilon or handle.
    # Original code doesn't explicitly handle denominator zero for the division itself.
    # It adds epsilon to vwap, which is not for this division.
    return (numerator.div(denominator.replace(0, 1e-6)) # Added replace(0,1e-6) for safety
            .stack('ticker')
            .swaplevel())

def alpha058(v, wvap, sector): # wvap is likely vwap
    """
    Alpha Factor #58: (-1 * ts_rank(ts_weighted_mean(ts_corr(IndNeutralize(vwap, IndClass.sector), volume, 3), 7), 5))

    中文注释:
    入参:
        v: pandas DataFrame, 成交量。
        vwap: pandas DataFrame, 成交量加权平均价 (函数参数为 wvap，推测为 vwap)。
        sector: pandas Series/DataFrame, 股票对应的行业分类信息 (IndClass.sector)。
    适用性:
        截面/时序混合 (IndNeutralize, ts_corr, ts_weighted_mean, ts_rank)
    Alpha 含义:
        此 Alpha 关注行业中性化后的VWAP与成交量相关性的加权均值的时序排名。
        1. IndNeutralize(vwap, IndClass.sector): 对VWAP按行业进行中性化处理。
        2. ts_corr(IndNeutralized_vwap, volume, 3): 中性化VWAP与成交量在过去3日的相关性。
        3. ts_weighted_mean(..., 7): 对上述相关性进行7日加权移动平均。
        4. ts_rank(..., 5): 对加权平均相关性进行过去5日的时序排名。
        5. Alpha = -1 * ...
        该 Alpha 试图识别那些“行业中性VWAP-成交量相关性”的近期加权平均值在其自身近期历史中排名较低（ts_rank高，乘以-1后为负）的股票。
        【注意】: 此函数标为 `pass`，未实现。注释基于公式字符串。`IndClass.sector` 需在别处定义。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    pass

def alpha059(v, wvap, industry): # wvap is likely vwap
    """
    Alpha Factor #59: -ts_rank(ts_weighted_mean(ts_corr(IndNeutralize(vwap, IndClass.industry), volume, 4), 16), 8)

    中文注释:
    入参:
        v: pandas DataFrame, 成交量。
        vwap: pandas DataFrame, 成交量加权平均价 (函数参数为 wvap，推测为 vwap)。
        industry: pandas Series/DataFrame, 股票对应的行业分类信息 (IndClass.industry)。
    适用性:
        截面/时序混合 (IndNeutralize, ts_corr, ts_weighted_mean, ts_rank)
    Alpha 含义:
        与 Alpha058 结构类似，但参数不同：
        1. IndNeutralize(vwap, IndClass.industry): 对VWAP按行业 (IndClass.industry) 进行中性化。
        2. ts_corr(..., volume, 4): 中性化VWAP与成交量在过去4日的相关性。
        3. ts_weighted_mean(..., 16): 对相关性进行16日加权移动平均。
        4. ts_rank(..., 8): 对加权平均相关性进行过去8日的时序排名。
        5. Alpha = -1 * ... (公式中没有-1，但通常这类因子会有方向性，这里按公式字符串)
           Alpha = -ts_rank(...)
        该 Alpha 逻辑与058相似，但使用了更细的行业分类 (industry vs sector) 以及不同的时间窗口。
        【注意】: 此函数标为 `pass`，未实现。注释基于公式字符串。`IndClass.industry` 需在别处定义。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    pass

def alpha060(l, h, c, v):
    """
    Alpha Factor #60: -((2 * scale(rank(((((close - low) - (high - close)) / (high - low + 1e-5)) * volume)))) - scale(rank(ts_argmax(close, 10))))

    中文注释:
    入参:
        l: pandas DataFrame, 最低价。
        h: pandas DataFrame, 最高价。
        c: pandas DataFrame, 收盘价。
        v: pandas DataFrame, 成交量。
    适用性:
        截面 Alpha (使用了 scale, rank, ts_argmax)
    Alpha 含义:
        此 Alpha 由两部分构成，衡量了价格在日内振幅中的位置（成交量加权）与近期价格高点出现时间。
        Part 1: 2 * scale(rank( ( ( (close - low) - (high - close) ) / (high - low + 1e-5) ) * volume ))
            - inner_ratio = ((close - low) - (high - close)) / (high - low + 1e-5)
                          = (2*close - low - high) / (high - low + 1e-5)
              这个比率衡量收盘价相对于当日振幅中点的位置。如果 c 是中点，比率为0。如果 c 接近 h，比率接近1。如果 c 接近 l，比率接近-1。
            - (inner_ratio * volume): 用成交量加权。
            - rank(...): 对加权后的结果进行排名。
            - scale(...): 标准化。
            - 2 * ...: 乘以2放大。
        Part 2: scale(rank(ts_argmax(close, 10)))
            - ts_argmax(close, 10): 过去10日收盘价最高点出现的时间（天数）。
            - rank(...): 对其排名。
            - scale(...): 标准化。
        Alpha = -(Part1 - Part2) = Part2 - Part1。
        它试图寻找那些近期高点刚出现（Part2的rank值小，scale后可能为负），并且成交量加权的日内价格位置指标（Part1）也表现出特定模式的股票。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    # inner_ratio_numerator = (c.sub(l)).sub(h.sub(c)) which is c - l - h + c = 2c - l - h
    # inner_ratio_denominator = h.sub(l).replace(0, 1e-5) # Added replace for safety, original has 1e-5
    # The code has: c.mul(2).sub(l).sub(h) which is 2c - l - h. Correct.
    # Denom: h.sub(l).replace(0, 1e-5). Correct.
    part1_val = c.mul(2).sub(l).sub(h).div(h.sub(l).replace(0, 1e-5)).mul(v)
    part1 = scale(rank(part1_val)).mul(2)
    part2 = scale(rank(ts_argmax(c, 10)))

    return (part1.sub(part2) # -(P1 - P2) = P2 - P1. Code: P1.sub(P2).mul(-1) which is P2-P1.
            .mul(-1) # This is (Part2 - Part1)
            .stack('ticker')
            .swaplevel())

def alpha061(v, vwap):
    """
    Alpha Factor #61: rank((vwap - ts_min(vwap, 16))) < rank(ts_corr(vwap, adv180, 17))
    (adv180 is 180-day average volume, typically calculated from v)

    中文注释:
    入参:
        v: pandas DataFrame, 成交量 (用于计算 adv180)。
        vwap: pandas DataFrame, 成交量加权平均价。
    适用性:
        截面 Alpha (使用了 rank, ts_corr, ts_min)
    Alpha 含义:
        此 Alpha 比较了两个排名的相对大小。
        Part 1: rank(vwap - ts_min(vwap, 16))
            - ts_min(vwap, 16): 过去16日VWAP的最小值。
            - vwap - ts_min(vwap, 16): 当前VWAP相对于近期VWAP低点的回升幅度。
            - rank(...): 对此回升幅度进行排名。
        Part 2: rank(ts_corr(vwap, adv180, 17))
            - adv180: 180日平均成交量 (代码中用 ts_mean(v, 180) 实现)。
            - ts_corr(vwap, adv180, 17): VWAP与180日均量在过去17日的相关性。
            - rank(...): 对此相关性进行排名。
        Alpha = (Part 1 < Part 2)。结果为布尔值 (True/False)，代码中转为整数 (1/0)。
        它试图寻找那些“VWAP从近期低点回升的幅度排名”小于“VWAP与长期均量相关性排名”的股票。
    出参:
        pandas Series, 计算得到的 Alpha 值 (1 或 0)。索引为 (日期, 股票代码) 的 MultiIndex。
        (代码中使用 .astype(int) 将布尔结果转为1/0)
    """
    adv180 = ts_mean(v, 180) # Calculate adv180 from v
    # Formula uses window 17 for corr, code uses 18. Using code's window.
    part1 = rank(vwap.sub(ts_min(vwap, 16)))
    part2 = rank(ts_corr(vwap, adv180, 18)) # Code uses 18, formula 17

    return (part1.lt(part2)
            .astype(int)
            .stack('ticker')
            .swaplevel())

def alpha062(o, h, l, vwap, adv20):
    """
    Alpha Factor #62: ((rank(ts_corr(vwap, ts_sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
    (adv20 is 20-day average volume)

    中文注释:
    入参:
        o: pandas DataFrame, 开盘价。
        h: pandas DataFrame, 最高价。
        l: pandas DataFrame, 最低价。
        vwap: pandas DataFrame, 成交量加权平均价。
        adv20: pandas DataFrame, 20日平均成交量。
    适用性:
        截面 Alpha (全部是 rank 操作的比较)
    Alpha 含义:
        此 Alpha 结构复杂，比较了两个复杂排名指标的大小，然后乘以-1。
        Let Cond1 = rank(ts_corr(vwap, ts_sum(adv20, 22), 9)) (代码中使用整数窗口)
            - ts_sum(adv20, 22): 20日均量在过去22日的累积值。
            - ts_corr(vwap, ..., 9): VWAP与上述累积均量在过去9日的相关性排名。
        Let Cond2_inner_left = rank(open) + rank(open) = 2 * rank(open)
        Let Cond2_inner_right = rank((high + low) / 2) + rank(high)
        Let Cond2 = rank(Cond2_inner_left < Cond2_inner_right)
            - 比较“两倍开盘价排名”是否小于“中间价排名与最高价排名之和”，然后对这个布尔结果再排名。
        Alpha = (Cond1 < Cond2) * -1。
        它试图寻找满足特定量价相关性排名条件与多种价格特征组合排名条件的股票。
    出参:
        pandas Series, 计算得到的 Alpha 值 (-1, 0, 或 1, 取决于比较结果和乘法)。
        (布尔比较结果转为1/0后乘以-1，所以是 -1 或 0)
    """
    # ts_sum(adv20, 22.4101) -> ts_sum(adv20, 22) in code
    # ts_corr window 9.91009 -> 9 in code
    cond1 = rank(ts_corr(vwap, ts_sum(adv20, 22), 9))

    cond2_inner_left = rank(o).mul(2)
    cond2_inner_right = rank(h.add(l).div(2)).add(rank(h))
    cond2 = rank(cond2_inner_left.lt(cond2_inner_right)) # rank of a boolean series

    return (cond1.lt(cond2)
            .mul(-1) # Converts True to -1, False to 0
            .stack('ticker')
            .swaplevel())

def alpha063(v, wvap, industry): # wvap -> vwap, close is missing
    """
    Alpha Factor #63: ((rank(ts_weighted_mean(ts_delta(IndNeutralize(close, IndClass.industry), 2), 8)) - rank(ts_weighted_mean(ts_corr(((vwap * 0.318108) + (open * (1 - 0.318108))), ts_sum(adv180, 37), 13), 12))) * -1)
    (Requires: close, open, adv180 in addition to v, vwap, industry)

    中文注释:
    入参:
        v: pandas DataFrame, 成交量 (用于计算 adv180)。
        vwap: pandas DataFrame, 成交量加权平均价 (函数参数 wvap)。
        industry: pandas Series/DataFrame, 股票对应的行业分类信息 (IndClass.industry)。
        (隐含需要: close, open。 adv180 也需要从 v 计算得到)
    适用性:
        截面 Alpha (涉及 IndNeutralize, rank, ts_weighted_mean, ts_corr)
    Alpha 含义:
        此 Alpha 由两大部分的排名之差构成，然后乘以-1。
        Part 1: rank(ts_weighted_mean(ts_delta(IndNeutralize(close, IndClass.industry), 2), 8))
            - IndNeutralize(close, IndClass.industry): 收盘价按行业中性化。
            - ts_delta(..., 2): 中性化收盘价的2日变化。
            - ts_weighted_mean(..., 8): 上述变化的8日加权均值。
            - rank(...): 对其排名。
        Part 2: rank(ts_weighted_mean(ts_corr( (vwap*w1 + open*(1-w1)), ts_sum(adv180, 37), 13), 12)) where w1=0.318108
            - price_combo = vwap * 0.318108 + open * (1 - 0.318108): VWAP和开盘价的加权组合。
            - ts_sum(adv180, 37): 180日均量在过去37日的累积。
            - ts_corr(price_combo, sum_adv180, 13): 价格组合与累积均量在过去13日的相关性。
            - ts_weighted_mean(..., 12): 对相关性进行12日加权均值。
            - rank(...): 对其排名。
        Alpha = (Part1 - Part2) * -1 = Part2 - Part1。
        它比较了“行业中性价格动量的加权均值排名”和“特定价格组合与成交量累积相关性的加权均值排名”。
        【注意】: 此函数标为 `pass`，未实现。需要额外数据 (close, open)。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    pass

def alpha064(o, h, l, v, vwap):
    """
    Alpha Factor #64: ((rank(ts_corr(ts_sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054),ts_sum(adv120, 12.7054), 16.6208)) < rank(ts_delta(((((high + low) / 2) * 0.178404) + (vwap * (1 -0.178404))), 3.69741))) * -1)
    (adv120 is 120-day average volume)

    中文注释:
    入参:
        o: pandas DataFrame, 开盘价。
        h: pandas DataFrame, 最高价。
        l: pandas DataFrame, 最低价。
        v: pandas DataFrame, 成交量 (用于计算 adv120)。
        vwap: pandas DataFrame, 成交量加权平均价。
    适用性:
        截面 Alpha (全部是 rank 操作的比较)
    Alpha 含义:
        此 Alpha 比较两个复杂排名的大小，然后乘以-1。权重 w = 0.178404。
        Cond1_price_combo = open * w + low * (1-w)
        Cond1_adv_combo = ts_sum(adv120, 12.7) (adv120 = 120日均量, 代码中用 ts_mean(v,120))
        Cond1 = rank(ts_corr(ts_sum(Cond1_price_combo, 12.7), Cond1_adv_combo, 16.6))

        Cond2_price_combo = ((high + low) / 2) * w + vwap * (1-w)
        Cond2 = rank(ts_delta(Cond2_price_combo, 3.7))

        (代码中使用整数窗口: 12, 12, 16 for Cond1; 3 for Cond2_delta)
        Alpha = (Cond1 < Cond2) * -1。
        它寻找满足特定加权价格组合与加权成交量组合的相关性排名条件，相对于另一种加权价格组合的动量排名条件的股票。
    出参:
        pandas Series, 计算得到的 Alpha 值 (-1 或 0)。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    w = 0.178404
    adv120 = ts_mean(v, 120)

    cond1_price_combo_sum = ts_sum(o.mul(w).add(l.mul(1 - w)), 12) # win 12.7 -> 12
    cond1_adv_combo_sum = ts_sum(adv120, 12) # win 12.7 -> 12
    cond1 = rank(ts_corr(cond1_price_combo_sum, cond1_adv_combo_sum, 16)) # win 16.6 -> 16

    cond2_price_combo = (h.add(l).div(2)).mul(w).add(vwap.mul(1 - w))
    cond2 = rank(ts_delta(cond2_price_combo, 3)) # win 3.69 -> 3

    return (cond1.lt(cond2)
            .mul(-1)
            .stack('ticker')
            .swaplevel())

def alpha065(o, v, vwap):
    """
    Alpha Factor #65: ((rank(ts_corr(((open * 0.00817205) + (vwap * (1 - 0.00817205))), ts_sum(adv60,8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)
    (adv60 is 60-day average volume)

    中文注释:
    入参:
        o: pandas DataFrame, 开盘价。
        v: pandas DataFrame, 成交量 (用于计算 adv60)。
        vwap: pandas DataFrame, 成交量加权平均价。
    适用性:
        截面 Alpha (全部是 rank 操作的比较)
    Alpha 含义:
        此 Alpha 比较两个排名的大小，然后乘以-1。权重 w = 0.00817205。
        Cond1_price_combo = open * w + vwap * (1-w)
        Cond1_adv_combo = ts_sum(adv60, 8.7) (adv60 = 60日均量, 代码中用 ts_mean(ts_mean(v,60),9) )
                                            (ts_sum(adv60, 8.69) vs code's ts_mean(adv60,9) - different concepts)
                                            Let's assume code's version for adv part: ts_mean(adv60, 9)
        Cond1 = rank(ts_corr(Cond1_price_combo, ts_mean(adv60, 9), 6.4))

        Cond2 = rank(open - ts_min(open, 13.6))

        (代码中使用整数窗口: adv60 calc from v; ts_mean for adv part window 9; corr window 6; ts_min window 13)
        Alpha = (Cond1 < Cond2) * -1。
        它寻找满足特定加权价格组合 (主要是VWAP) 与处理后均量的相关性排名条件，相对于开盘价从近期低点回升幅度的排名条件的股票。
    出参:
        pandas Series, 计算得到的 Alpha 值 (-1 或 0)。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    w = 0.00817205
    adv60 = ts_mean(v, 60)

    cond1_price_combo = o.mul(w).add(vwap.mul(1 - w))
    # Formula: ts_sum(adv60, 8.6911)
    # Code: ts_mean(ts_mean(v, 60), 9) which is ts_mean(adv60, 9)
    cond1_adv_processed = ts_mean(adv60, 9) # Using code's version
    cond1 = rank(ts_corr(cond1_price_combo, cond1_adv_processed, 6)) # win 6.4 -> 6

    cond2 = rank(o.sub(ts_min(o, 13))) # win 13.6 -> 13

    return (cond1.lt(cond2)
            .mul(-1)
            .stack('ticker')
            .swaplevel())

def alpha066(h, l, o, vwap): # Parameter v is missing for adv if needed. Assuming no adv here.
    """
    Alpha Factor #66: ((rank(ts_weighted_mean(ts_delta(vwap, 3.51013), 7.23052)) + ts_rank(ts_weighted_mean(((((low* 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
    (Note: low * 0.96633 + low * (1-0.96633) simply equals low. This might be a simplified version or a placeholder in the formula string.)

    中文注释: (基于代码实现，其中 low*w + low*(1-w) = low)
    入参:
        h: pandas DataFrame, 最高价。
        l: pandas DataFrame, 最低价。
        o: pandas DataFrame, 开盘价。
        vwap: pandas DataFrame, 成交量加权平均价。
    适用性:
        截面/时序混合 (rank, ts_weighted_mean, ts_delta, ts_rank)
    Alpha 含义:
        此 Alpha 由两大部分相加，然后乘以-1。
        Part 1: rank(ts_weighted_mean(ts_delta(vwap, 3.5), 7.2)) (代码窗口: 4, 7)
            - ts_delta(vwap, 4): VWAP的4日变化。
            - ts_weighted_mean(..., 7): 上述变化的7日加权均值。
            - rank(...): 对其排名。
        Part 2: ts_rank(ts_weighted_mean( (low - vwap) / (open - (high+low)/2 + 1e-3) , 11.4), 6.7) (代码窗口: 11, 7)
            - numerator = low - vwap
            - denominator = open - (high+low)/2  (开盘价与当日中间价的差)
            - ratio = numerator / (denominator + 1e-3) (防止除零)
            - ts_weighted_mean(ratio, 11): 上述比率的11日加权均值。
            - ts_rank(..., 7): 对加权均值进行7日时序排名。
        Alpha = (Part1 + Part2) * -1。
        它结合了VWAP动量的加权均值排名，与一个衡量最低价相对VWAP的偏离（并用开盘价与中间价的偏离进行调整）的复杂指标的时序排名。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    # w = 0.96633 # This weight makes low*w + low*(1-w) = low
    # So, (low*w + low*(1-w)) - vwap simplifies to low - vwap

    part1 = rank(ts_weighted_mean(ts_delta(vwap, 4), 7)) # Windows from code

    numerator_p2 = l.sub(vwap)
    denominator_p2 = o.sub(h.add(l).div(2)).add(1e-3) # Added epsilon
    ratio_p2 = numerator_p2.div(denominator_p2)
    part2 = ts_rank(ts_weighted_mean(ratio_p2, 11), 7) # Windows from code

    return (part1.add(part2)
            .mul(-1)
            .stack('ticker')
            .swaplevel())

def alpha067(h, v, sector, subindustry): # Requires vwap, adv20
    """
    Alpha Factor #67: (power(rank((high - ts_min(high, 2.14593))), rank(ts_corr(IndNeutralize(vwap,IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)
    (Requires vwap, adv20 in addition to listed params. adv20 from v)

    中文注释:
    入参:
        h: pandas DataFrame, 最高价。
        v: pandas DataFrame, 成交量 (用于计算 adv20)。
        sector: pandas Series/DataFrame, 股票对应的行业分类信息 (IndClass.sector)。
        subindustry: pandas Series/DataFrame, 股票对应的子行业分类信息 (IndClass.subindustry)。
        (隐含需要: vwap)
    适用性:
        截面 Alpha (IndNeutralize, rank, ts_corr, power)
    Alpha 含义:
        此 Alpha 计算两个排名项的幂运算，然后乘以-1。
        Base: rank(high - ts_min(high, 2.1)) (代码窗口近似为2)
            - high - ts_min(high, 2): 当前最高价与近2日最低价之差（衡量价格从近期低点反弹的幅度）。
            - rank(...): 对其排名。
        Exponent: rank(ts_corr(IndNeutralize(vwap, IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6))
            - IndNeutralize(vwap, IndClass.sector): VWAP按行业中性化。
            - IndNeutralize(adv20, IndClass.subindustry): 20日均量按子行业中性化。
            - ts_corr(..., ..., 6): 上述两个中性化指标在过去6日的相关性。
            - rank(...): 对此相关性排名。
        Alpha = (Base ^ Exponent) * -1。
        它捕捉了价格反弹幅度的排名，与“行业中性VWAP”和“子行业中性均量”的相关性排名之间的幂律关系。
        【注意】: 此函数标为 `pass`，未实现。需要额外数据 (vwap)。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    pass

def alpha068(c, h, l, v): # Requires adv15
    """
    Alpha Factor #68: ((ts_rank(ts_corr(rank(high), rank(adv15), 8.91644), 13.9333) < rank(ts_delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
    (adv15 is 15-day average volume, from v)

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。
        h: pandas DataFrame, 最高价。
        l: pandas DataFrame, 最低价。
        v: pandas DataFrame, 成交量 (用于计算 adv15)。
    适用性:
        截面/时序混合 (ts_rank, ts_corr, rank, ts_delta)
    Alpha 含义:
        此 Alpha 比较两个排名的大小，然后乘以-1。权重 w = 0.518371。
        Cond1 = ts_rank(ts_corr(rank(high), rank(adv15), 8.9), 13.9) (代码窗口: 9, 14)
            - rank(high): 最高价排名。
            - rank(adv15): 15日均量排名。
            - ts_corr(..., ..., 9): 两者排名在过去9日的相关性。
            - ts_rank(..., 14): 对相关性进行14日时序排名。
        Cond2_price_combo = close * w + low * (1-w)
        Cond2 = rank(ts_delta(Cond2_price_combo, 1.06)) (代码窗口: 1)
            - 加权价格组合的1日变化排名。
        Alpha = (Cond1 < Cond2) * -1。
        它寻找那些“最高价排名与均量排名的相关性的时序排名”小于“特定价格组合动量排名”的股票。
    出参:
        pandas Series, 计算得到的 Alpha 值 (-1 或 0)。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    w = 0.518371
    adv15 = ts_mean(v, 15)

    cond1 = ts_rank(ts_corr(rank(h), rank(adv15), 9), 14) # Windows from code

    cond2_price_combo = c.mul(w).add(l.mul(1 - w))
    cond2 = rank(ts_delta(cond2_price_combo, 1)) # Window from code

    return (cond1.lt(cond2)
            .mul(-1)
            .stack('ticker')
            .swaplevel())

def alpha069(c, vwap, industry): # Requires adv20
    """
    Alpha Factor #69: ((power(rank(ts_max(ts_delta(IndNeutralize(vwap, IndClass.industry), 2.72412),4.79344)), Ts_Rank(ts_corr(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416),9.0615))) * -1)
    (adv20 from v)

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。
        vwap: pandas DataFrame, 成交量加权平均价。
        industry: pandas Series/DataFrame, 股票对应的行业分类信息 (IndClass.industry)。
        (隐含需要: v for adv20)
    适用性:
        截面/时序混合 (IndNeutralize, rank, ts_max, ts_delta, Ts_Rank, ts_corr, power)
    Alpha 含义:
        此 Alpha 计算两个复杂项的幂运算，然后乘以-1。权重 w = 0.490655。
        Base: rank(ts_max(ts_delta(IndNeutralize(vwap, IndClass.industry), 2.7), 4.8))
            - IndNeutralize(vwap, IndClass.industry): VWAP按行业中性化。
            - ts_delta(..., 2.7): 中性化VWAP的2.7日变化。
            - ts_max(..., 4.8): 上述变化在过去4.8日的最大值。
            - rank(...): 对其排名。
        Exponent: Ts_Rank(ts_corr((close*w + vwap*(1-w)), adv20, 4.9), 9.0)
            - price_combo = close*w + vwap*(1-w): 收盘价和VWAP的加权组合。
            - ts_corr(price_combo, adv20, 4.9): 价格组合与20日均量的4.9日相关性。
            - Ts_Rank(..., 9.0): 对相关性进行9日时序排名。
        Alpha = (Base ^ Exponent) * -1。
        它捕捉了“行业中性VWAP动量的近期最大值排名”与“特定价格组合与均量相关性的时序排名”之间的幂律关系。
        【注意】: 此函数标为 `pass`，未实现。需要额外数据 (v for adv20)。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    pass

# alpha070 is missing in the provided file. Will proceed to alpha071.

def alpha071(c, l, o, v, vwap): # adv180 from v
    """max(ts_rank(ts_weighted_mean(ts_corr(ts_rank(close, 3.43976), ts_rank(adv180,12.0647), 18.0175), 4.20501), 15.6948), 
            ts_rank(ts_weighted_mean((rank(((low + open) - (vwap +vwap)))^2), 16.4662), 4.4388))"""
    # adv180 from v
    adv180 = ts_mean(v, 180)
    s1 = (ts_rank(ts_weighted_mean(ts_corr(ts_rank(c, 3), # win 3.43 -> 3
                                           ts_rank(adv180, 12), 18), 4), 16)) # win 12.06 -> 12, 18.01 -> 18, 4.2 -> 4, 15.69 -> 16

    # Formula: rank(((low + open) - (vwap +vwap)))^2  -- this means rank(A)^2
    # Code: rank( ((low+open) - 2*vwap)^2 )
    s2_inner_val = l.add(o).sub(vwap.mul(2))
    s2 = (ts_rank(ts_weighted_mean(rank(s2_inner_val.pow(2)), 16), 4)) # win 16.46 -> 16, 4.43 -> 4

    return (s1.where(s1 > s2, s2) # max(s1, s2)
            .stack('ticker')
            .swaplevel())

def alpha072(h, l, v, vwap): # adv40 from v
    """
    Alpha Factor #072: (rank(ts_weighted_mean(ts_corr(((high + low) / 2), adv40, 8.93345), 10.1519)) / rank(ts_weighted_mean(ts_corr(ts_rank(vwap, 3.72469), ts_rank(volume, 18.5188), 6.86671), 2.95011)))

    中文注释:
    入参:
        h: pandas DataFrame, 最高价。
        l: pandas DataFrame, 最低价。
        v: pandas DataFrame, 成交量 (用于计算 adv40)。
        vwap: pandas DataFrame, 成交量加权平均价。
    适用性:
        截面/时序混合 (rank, ts_weighted_mean, ts_corr, ts_rank)
    Alpha 含义:
        此 Alpha 是两个复杂排名指标的比值。
        Numerator: rank(ts_weighted_mean(ts_corr((high+low)/2, adv40, 8.9), 10.1))
            - (high+low)/2: 当日中间价。
            - adv40: 40日均量。
            - ts_corr(mid_price, adv40, 9): 中间价与40日均量在过去9日的相关性。(代码窗口: 9)
            - ts_weighted_mean(..., 10): 上述相关性的10日加权均值。(代码窗口: 10)
            - rank(...): 对其排名。
        Denominator: rank(ts_weighted_mean(ts_corr(ts_rank(vwap, 3.7), ts_rank(volume, 18.5), 6.8), 2.9))
            - ts_rank(vwap, 4): VWAP的4日时序排名。(代码窗口: 3)
            - ts_rank(volume, 18): 成交量的18日时序排名。(代码窗口: 18)
            - ts_corr(..., ..., 7): 上述两个时序排名的7日相关性。(代码窗口: 6)
            - ts_weighted_mean(..., 3): 对相关性进行3日加权均值。(代码窗口: 2)
            - rank(...): 对其排名。
        Alpha = Numerator / Denominator。
        它比较了“价格中枢与均量相关性的加权均值排名”和“VWAP时序排名与成交量时序排名的相关性的加权均值排名”。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    adv40 = ts_mean(v, 40)

    num = rank(ts_weighted_mean(ts_corr(h.add(l).div(2), adv40, 9), 10)) # Windows from code
    den = rank(ts_weighted_mean(ts_corr(ts_rank(vwap, 3), ts_rank(v, 18), 6), 2)) # Windows from code

    return (num.div(den.replace(0, 1e-6)) # Avoid division by zero
            .stack('ticker')
            .swaplevel())

def alpha073(l, o, vwap): # v is implicitly needed if advXXX were used, but not in this formula.
    """
    Alpha Factor #073: (max(rank(ts_weighted_mean(ts_delta(vwap, 4.72775), 2.91864)), ts_rank(ts_weighted_mean(((ts_delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open *0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)

    中文注释:
    入参:
        l: pandas DataFrame, 最低价。
        o: pandas DataFrame, 开盘价。
        vwap: pandas DataFrame, 成交量加权平均价。
    适用性:
        截面/时序混合 (rank, ts_weighted_mean, ts_delta, ts_rank)
    Alpha 含义:
        此 Alpha 取两个复杂指标中的较大者，然后乘以-1。权重 w = 0.147155。
        Term1: rank(ts_weighted_mean(ts_delta(vwap, 4.7), 2.9)) (代码窗口: 5, 3)
            - ts_delta(vwap, 5): VWAP的5日变化。
            - ts_weighted_mean(..., 3): 上述变化的3日加权均值。
            - rank(...): 对其排名。
        Term2: ts_rank(ts_weighted_mean( (ts_delta(price_combo, 2.0) / price_combo) * -1, 3.3), 16.7) (代码窗口: 2, 3, 16)
            - price_combo = open * w + low * (1-w): 开盘价和最低价的加权组合。
            - relative_delta = ts_delta(price_combo, 2) / price_combo: 加权价格组合的2日相对变化率。
            - ts_weighted_mean(relative_delta * -1, 3): 上述相对变化率的负值的3日加权均值。
            - ts_rank(..., 16): 对其进行16日时序排名。
        Alpha = max(Term1, Term2) * -1。
        它结合了VWAP动量均值的排名，与一个衡量特定价格组合相对变化率的复杂指标的时序排名。
        `print(s2)` 是调试信息。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    w = 0.147155
    s1 = rank(ts_weighted_mean(ts_delta(vwap, 5), 3)) # Windows from code

    price_combo = o.mul(w).add(l.mul(1 - w))
    # Formula: (delta(price_combo, 2) / price_combo) * -1
    # Code: delta(price_combo,2).div(price_combo.mul(-1)) which is (delta / -price_combo) = -(delta/price_combo)
    # This matches the formula.
    relative_delta_neg = ts_delta(price_combo, 2).div(price_combo.mul(-1))
    s2 = ts_rank(ts_weighted_mean(relative_delta_neg, 3), 16) # Windows from code

    print(s2) # Debug print statement from original code
    return (s1.where(s1 > s2, s2) # max(s1,s2)
            .mul(-1)
            .stack('ticker')
            .swaplevel())

def alpha074(c, h, v, vwap): # adv30 from v
    """
    Alpha Factor #074: ((rank(ts_corr(close, ts_sum(adv30, 37.4843), 15.1365)) < rank(ts_corr(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791)))* -1)

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。
        h: pandas DataFrame, 最高价。
        v: pandas DataFrame, 成交量 (用于计算 adv30)。
        vwap: pandas DataFrame, 成交量加权平均价。
    适用性:
        截面 Alpha (rank, ts_corr)
    Alpha 含义:
        此 Alpha 比较两个相关性排名的大小，然后乘以-1。权重 w = 0.0261661。
        Cond1: rank(ts_corr(close, ts_sum(adv30, 37.5), 15.1)) (代码窗口: 37, 15)
            - adv30: 30日均量。
            - ts_sum(adv30, 37): 30日均量在过去37日的累积值。(代码中 ts_mean(ts_mean(v,30),37) 即 adv30 的37日均值)
            - ts_corr(close, processed_adv30, 15): 收盘价与处理后均量的15日相关性。
            - rank(...): 对其排名。
        Cond2_price_combo = high * w + vwap * (1-w)
        Cond2: rank(ts_corr(rank(Cond2_price_combo), rank(volume), 11.5)) (代码窗口: 11)
            - rank(Cond2_price_combo): 加权价格组合的排名。
            - rank(volume): 成交量排名。
            - ts_corr(..., ..., 11): 两个排名的11日相关性。
            - rank(...): 对其排名。
        Alpha = (Cond1 < Cond2) * -1。
        它比较了“收盘价与处理后均量的相关性排名”和“特定价格组合排名与成交量排名的相关性排名”。
    出参:
        pandas Series, 计算得到的 Alpha 值 (-1 或 0)。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    w = 0.0261661
    adv30 = ts_mean(v, 30)

    # Formula: ts_sum(adv30, 37.4843)
    # Code: ts_mean(ts_mean(v, 30), 37) which is ts_mean(adv30, 37)
    processed_adv30 = ts_mean(adv30, 37) # Using code's version
    cond1 = rank(ts_corr(c, processed_adv30, 15)) # Window 15.13 -> 15

    cond2_price_combo = h.mul(w).add(vwap.mul(1 - w))
    cond2 = rank(ts_corr(rank(cond2_price_combo), rank(v), 11)) # Window 11.47 -> 11

    return (cond1.lt(cond2)
            .mul(-1)
            .stack('ticker')
            .swaplevel())

def alpha075(l, v, vwap): # adv50 from v
    """
    Alpha Factor #075: (rank(ts_corr(vwap, volume, 4.24304)) < rank(ts_corr(rank(low), rank(adv50),12.4413)))

    中文注释:
    入参:
        l: pandas DataFrame, 最低价。
        v: pandas DataFrame, 成交量 (用于计算 adv50)。
        vwap: pandas DataFrame, 成交量加权平均价。
    适用性:
        截面 Alpha (rank, ts_corr)
    Alpha 含义:
        此 Alpha 比较两个相关性排名的大小。
        Cond1: rank(ts_corr(vwap, volume, 4.2)) (代码窗口: 4)
            - ts_corr(vwap, volume, 4): VWAP与成交量的4日相关性。
            - rank(...): 对其排名。
        Cond2: rank(ts_corr(rank(low), rank(adv50), 12.4)) (代码窗口: 12)
            - rank(low): 最低价排名。
            - adv50: 50日均量。
            - rank(adv50): 50日均量排名。
            - ts_corr(..., ..., 12): 最低价排名与均量排名的12日相关性。
            - rank(...): 对其排名。
        Alpha = (Cond1 < Cond2)。结果为布尔值，代码中转为整数 (1/0)。
        它比较了“VWAP与成交量的短期相关性排名”和“最低价排名与均量排名的稍长期相关性排名”。
    出参:
        pandas Series, 计算得到的 Alpha 值 (1 或 0)。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    adv50 = ts_mean(v, 50)

    cond1 = rank(ts_corr(vwap, v, 4)) # Window 4.24 -> 4
    cond2 = rank(ts_corr(rank(l), rank(adv50), 12)) # Window 12.44 -> 12

    return (cond1.lt(cond2)
            .astype(int)
            .stack('ticker')
            .swaplevel())

# alpha076 is missing in the file, skipping to alpha077

def alpha077(h, l, v, vwap): # adv40 from v
    """
    Alpha Factor #077: min(rank(ts_weighted_mean(((((high + low) / 2) + high) - (vwap + high)), 20.0451)), rank(ts_weighted_mean(ts_corr(((high + low) / 2), adv40, 3.1614), 5.64125)))
    Formula simplification: ( (h+l)/2 + h ) - (vwap + h) = (h+l)/2 - vwap

    中文注释:
    入参:
        h: pandas DataFrame, 最高价。
        l: pandas DataFrame, 最低价。
        v: pandas DataFrame, 成交量 (用于计算 adv40)。
        vwap: pandas DataFrame, 成交量加权平均价。
    适用性:
        截面/时序混合 (rank, ts_weighted_mean, ts_corr)
    Alpha 含义:
        此 Alpha 取两个复杂排名指标中的较小者。
        Term1: rank(ts_weighted_mean( (high+low)/2 - vwap , 20.0)) (代码窗口: 20)
            - (high+low)/2 - vwap: 当日价格中枢与VWAP的差。
            - ts_weighted_mean(..., 20): 上述差值的20日加权均值。
            - rank(...): 对其排名。
        Term2: rank(ts_weighted_mean(ts_corr((high+low)/2, adv40, 3.16), 5.64)) (代码窗口: 3, 5)
            - (high+low)/2: 当日价格中枢。
            - adv40: 40日均量。
            - ts_corr(mid_price, adv40, 3): 中间价与40日均量在过去3日的相关性。
            - ts_weighted_mean(..., 5): 上述相关性的5日加权均值。
            - rank(...): 对其排名。
        Alpha = min(Term1, Term2)。
        它结合了“价格中枢与VWAP偏离的加权均值排名”和“价格中枢与均量相关性的加权均值排名”，取两者中较小的一个。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    adv40 = ts_mean(v, 40)

    # Formula term simplifies: ( (h+l)/2 + h ) - (vwap + h) = (h+l)/2 - vwap
    term1_val = (h.add(l).div(2)).sub(vwap)
    s1 = rank(ts_weighted_mean(term1_val, 20)) # Window from code

    term2_corr_val = ts_corr(h.add(l).div(2), adv40, 3) # Window from code
    s2 = rank(ts_weighted_mean(term2_corr_val, 5)) # Window from code

    return (s1.where(s1 < s2, s2) # min(s1, s2)
            .stack('ticker')
            .swaplevel())

def alpha078(l, v, vwap): # adv40 from v
    """
    Alpha Factor #078: (rank(ts_corr(ts_sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428), ts_sum(adv40, 19.7428), 6.83313))^rank(ts_corr(rank(vwap), rank(volume), 5.77492)))

    中文注释:
    入参:
        l: pandas DataFrame, 最低价。
        v: pandas DataFrame, 成交量 (用于计算 adv40 和直接使用)。
        vwap: pandas DataFrame, 成交量加权平均价。
    适用性:
        截面 Alpha (rank, ts_corr, ts_sum, power)
    Alpha 含义:
        此 Alpha 是一个幂运算，底数和指数都是复杂的排名项。权重 w = 0.352233。
        Base: rank(ts_corr(ts_sum(low*w + vwap*(1-w), 19.7), ts_sum(adv40, 19.7), 6.8)) (代码窗口: 19,19,6)
            - price_combo_sum = ts_sum(low*w + vwap*(1-w), 19): 特定价格组合的19日累积和。
            - adv40_sum = ts_sum(adv40, 19): 40日均量的19日累积和。
            - ts_corr(price_combo_sum, adv40_sum, 6): 两者累积和的6日相关性。
            - rank(...): 对其排名。
        Exponent: rank(ts_corr(rank(vwap), rank(volume), 5.8)) (代码窗口: 5)
            - rank(vwap): VWAP排名。
            - rank(volume): 成交量排名。
            - ts_corr(rank_vwap, rank_vol, 5): VWAP排名与成交量排名的5日相关性。
            - rank(...): 对其排名。
        Alpha = Base ^ Exponent。
        它捕捉了“特定价格组合累积与均量累积的相关性排名”与“VWAP排名与成交量排名相关性的排名”之间的幂律关系。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    w = 0.352233
    adv40 = ts_mean(v, 40)

    base_price_combo_sum = ts_sum(l.mul(w).add(vwap.mul(1-w)), 19) # Window from code
    base_adv_sum = ts_sum(adv40, 19) # Window from code
    base = rank(ts_corr(base_price_combo_sum, base_adv_sum, 6)) # Window from code

    exponent = rank(ts_corr(rank(vwap), rank(v), 5)) # Window from code

    return (base.pow(exponent)
            .stack('ticker')
            .swaplevel())

def alpha079(o, v, sector): # Requires close, vwap, adv150
    """
    Alpha Factor #079: (rank(ts_delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))),IndClass.sector), 1.23438)) < rank(ts_corr(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150,9.18637), 14.6644)))
    (adv150 from v)

    中文注释:
    入参:
        o: pandas DataFrame, 开盘价。
        v: pandas DataFrame, 成交量 (用于计算 adv150)。
        sector: pandas Series/DataFrame, 股票对应的行业分类信息 (IndClass.sector)。
        (隐含需要: close, vwap)
    适用性:
        截面/时序混合 (IndNeutralize, rank, ts_delta, ts_corr, Ts_Rank)
    Alpha 含义:
        此 Alpha 比较两个排名的大小。权重 w = 0.60733。
        Cond1: rank(ts_delta(IndNeutralize(close*w + open*(1-w), IndClass.sector), 1.23))
            - price_combo = close*w + open*(1-w): 收盘价和开盘价的加权组合。
            - IndNeutralize(price_combo, IndClass.sector): 价格组合按行业中性化。
            - ts_delta(..., 1.23): 中性化价格组合的1.23日变化。
            - rank(...): 对其排名。
        Cond2: rank(ts_corr(Ts_Rank(vwap, 3.6), Ts_Rank(adv150, 9.2), 14.7))
            - Ts_Rank(vwap, 3.6): VWAP的3.6日时序排名。
            - Ts_Rank(adv150, 9.2): 150日均量的9.2日时序排名。
            - ts_corr(..., ..., 14.7): 两个时序排名的14.7日相关性。
            - rank(...): 对其排名。
        Alpha = (Cond1 < Cond2)。结果为布尔值，通常转为1/0。
        它比较了“行业中性价格组合动量的排名”和“VWAP时序排名与均量时序排名的相关性排名”。
        【注意】: 此函数标为 `pass`，未实现。需要额外数据 (close, vwap)。
    出参:
        pandas Series, 计算得到的 Alpha 值 (1 或 0)。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    pass

def alpha080(h, industry): # Requires open, v, adv10
    """
    Alpha Factor #080: ((power(rank(sign(ts_delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))),IndClass.industry), 4.04545))), ts_rank(ts_corr(high, adv10, 5.11456), 5.53756)) * -1)
    (adv10 from v)

    中文注释:
    入参:
        h: pandas DataFrame, 最高价。
        industry: pandas Series/DataFrame, 股票对应的行业分类信息 (IndClass.industry)。
        (隐含需要: open, v for adv10)
    适用性:
        截面/时序混合 (IndNeutralize, rank, sign, ts_delta, power, ts_rank, ts_corr)
    Alpha 含义:
        此 Alpha 计算两个复杂项的幂运算，然后乘以-1。权重 w = 0.868128。
        Base: rank(sign(ts_delta(IndNeutralize(open*w + high*(1-w), IndClass.industry), 4.0)))
            - price_combo = open*w + high*(1-w): 开盘价和最高价的加权组合。
            - IndNeutralize(price_combo, IndClass.industry): 价格组合按行业中性化。
            - ts_delta(..., 4.0): 中性化价格组合的4日变化。
            - sign(...): 取其符号。
            - rank(...): 对符号排名。
        Exponent: ts_rank(ts_corr(high, adv10, 5.1), 5.5)
            - adv10: 10日均量。
            - ts_corr(high, adv10, 5.1): 最高价与10日均量的5.1日相关性。
            - ts_rank(..., 5.5): 对相关性进行5.5日时序排名。
        Alpha = (Base ^ Exponent) * -1。
        它捕捉了“行业中性价格组合的动量方向排名”与“最高价与均量相关性的时序排名”之间的幂律关系。
        【注意】: 此函数标为 `pass`，未实现。需要额外数据 (open, v for adv10)。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    pass

def alpha081(v, vwap): # adv10 from v
    """
    Alpha Factor #081: -(rank(log(ts_product(rank((rank(ts_corr(vwap, ts_sum(adv10, 49.6054),8.47743))^4)), 14.9655))) < rank(ts_corr(rank(vwap), rank(volume), 5.07914)))

    中文注释:
    入参:
        v: pandas DataFrame, 成交量 (用于计算 adv10 和直接使用)。
        vwap: pandas DataFrame, 成交量加权平均价。
    适用性:
        截面 Alpha (rank, log, ts_product, ts_corr, ts_sum, power)
    Alpha 含义:
        此 Alpha 是一个条件型因子，比较两个复杂排名的大小，然后取负。
        Cond1: rank(log(ts_product(rank((rank(ts_corr(vwap, ts_sum(adv10, 49.6), 8.5)))^4), 15.0)))
            (代码窗口: adv10 from v; sum_adv10 win 50; corr win 8; product win 15)
            - adv10: 10日均量。
            - ts_sum(adv10, 50): 10日均量的50日累积。
            - ts_corr(vwap, sum_adv10, 8): VWAP与累积均量的8日相关性。
            - rank(...): 对相关性排名。
            - (...)^4: 上述排名的4次方。
            - rank(...): 对4次方结果再排名。
            - ts_product(..., 15): 对上述排名结果进行15日连乘。
            - log(...): 取对数。
            - rank(...): 对对数结果排名。
        Cond2: rank(ts_corr(rank(vwap), rank(volume), 5.1)) (代码窗口: 5)
            - rank(vwap): VWAP排名。
            - rank(volume): 当日成交量排名。
            - ts_corr(..., ..., 5): 两者排名的5日相关性。
            - rank(...): 对其排名。
        Alpha = -(Cond1 < Cond2)。结果为布尔值取负，即 -1 或 0。
        它比较了一个非常复杂的VWAP与累积均量相关的指标的排名，和另一个VWAP排名与成交量排名相关的指标的排名。
    出参:
        pandas Series, 计算得到的 Alpha 值 (-1 或 0)。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    adv10 = ts_mean(v, 10)

    # Cond1 calculation based on code structure
    corr_val = ts_corr(vwap, ts_sum(adv10, 50), 8)
    rank_corr_pow4 = rank(rank(corr_val).pow(4))
    product_val = ts_product(rank_corr_pow4, 15)
    log_product_val = log(product_val) # log can produce -inf if product_val is 0 or very small
    cond1 = rank(log_product_val.replace(-np.inf, np.nan)) # Replace -inf with NaN before ranking

    # Cond2 calculation
    cond2 = rank(ts_corr(rank(vwap), rank(v), 5))

    return (cond1.lt(cond2)
            .mul(-1)
            .stack('ticker')
            .swaplevel())

def alpha082(o, v, sector): # (open * 0.634196) +(open * (1 - 0.634196)) simplifies to open.
    """
    Alpha Factor #082: (min(rank(ts_weighted_mean(ts_delta(open, 1.46063), 14.8717)), ts_rank(ts_weighted_mean(ts_corr(IndNeutralize(volume, IndClass.sector), open, 17.4842), 6.92131), 13.4283)) * -1)
    
    中文注释:
    入参:
        o: pandas DataFrame, 开盘价。
        v: pandas DataFrame, 成交量。
        sector: pandas Series/DataFrame, 股票对应的行业分类信息 (IndClass.sector)。
    适用性:
        截面/时序混合 (rank, ts_weighted_mean, ts_delta, ts_rank, ts_corr, IndNeutralize)
    Alpha 含义:
        此 Alpha 取两个复杂项中的较小者，然后乘以-1。
        Term1: rank(ts_weighted_mean(ts_delta(open, 1.46), 14.87)) (代码窗口近似: 1, 15)
            - ts_delta(open, 1): 开盘价的1日变化。
            - ts_weighted_mean(..., 15): 上述变化的15日加权均值。
            - rank(...): 对其排名。
        Term2: ts_rank(ts_weighted_mean(ts_corr(IndNeutralize(volume, IndClass.sector), open, 17.48), 6.92), 13.43)
               (代码窗口近似: corr win 17, mean win 7, rank win 13)
            - IndNeutralize(volume, IndClass.sector): 成交量按行业中性化。
            - (open * 0.63 + open * 0.37) = open.
            - ts_corr(Ind_volume, open, 17): 中性化成交量与开盘价的17日相关性。
            - ts_weighted_mean(..., 7): 上述相关性的7日加权均值。
            - ts_rank(..., 13): 对加权均值进行13日时序排名。
        Alpha = min(Term1, Term2) * -1。
        它结合了“开盘价动量加权均值的排名”和“行业中性成交量与开盘价相关性的加权均值的时序排名”。
        【注意】: 此函数标为 `pass`，未实现。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    pass

def alpha083(c, h, l, v, vwap):
    """
    Alpha Factor #083: (rank(ts_lag((high - low) / ts_mean(close, 5), 2)) * rank(rank(volume)) / (((high - low) / ts_mean(close, 5) / (vwap - close))))

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。
        h: pandas DataFrame, 最高价。
        l: pandas DataFrame, 最低价。
        v: pandas DataFrame, 成交量。
        vwap: pandas DataFrame, 成交量加权平均价。
    适用性:
        截面 Alpha (rank, ts_lag, ts_mean)
    Alpha 含义:
        此 Alpha 结构为 A*B / (C/D) = A*B*D / C。
        Let S = (high - low) / ts_mean(close, 5) (日振幅相对于5日均价的比率，衡量相对波动幅度)
        A = rank(ts_lag(S, 2)): S指标滞后2日的排名。
        B = rank(rank(volume)): 成交量的二次排名（强成交量信号）。
        C = S (当前的相对波动幅度)
        D = vwap - close (VWAP与收盘价的差，代码中用 vwap.sub(c).add(1e-3) 防止除零)
        Alpha = rank( A * B * D / C )，最后对整体再做一次排名 (rank(rank(...)))。

        它试图捕捉那些：
        - 前期相对波动幅度 (S) 排名高 (A)。
        - 当前成交量非常活跃 (B)。
        - 当前VWAP高于收盘价 (D > 0)。
        - 当前相对波动幅度 (C) 相对较小（使得 D/C 较大）。
        的股票，并对这个组合指标进行排名。
        `replace((np.inf, -np.inf), np.nan)` 用于处理潜在的无穷大值。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。包含 NaN 值。
    """
    s = h.sub(l).div(ts_mean(c, 5)) # Relative volatility S

    # Numerator part of the inner rank: rank(ts_lag(s, 2)) * rank(rank(v))
    # Denominator part of the inner rank: s / (vwap - close + eps)
    # So the full inner term before the outermost rank is:
    # (rank(ts_lag(s,2)) * rank(rank(v))) * (vwap - close + eps) / s

    # Code structure: rank( ( (rank(ts_lag(s,2)) * rank(rank(v))) / s ) / (vwap-c+eps) )  <- This is not A*B*D/C
    # Code is: rank ( (A*B/C) / (1/D) ) essentially if D was 1/(vwap-c)
    # Code is: rank( rank(ts_lag(s,2)).mul(rank(rank(v))).div(s).div(vwap.sub(c).add(1e-3)) )
    # This means: rank ( (A*B/C) * (1/(vwap-c+eps)))
    # Which is rank ( A*B / (C*(vwap-c+eps)) )
    # Let's comment based on the code's calculation.

    term_A = rank(ts_lag(s, 2))
    term_B = rank(rank(v))
    term_C = s
    term_D_inv = vwap.sub(c).add(1e-3) # This is (vwap - close + eps)

    combined_val = term_A.mul(term_B).div(term_C.replace(0,1e-6)).div(term_D_inv.replace(0,1e-6)) # Avoid div by zero for s and D_inv

    return (rank(combined_val) # Outermost rank
            .stack('ticker')
            .swaplevel()
            .replace((np.inf, -np.inf), np.nan))

def alpha084(c, vwap):
    """
    Alpha Factor #084: power(ts_rank((vwap - ts_max(vwap, 15.3217)), 20.7127), ts_delta(close,4.96796))
    (Final result is ranked in code, which is not in the formula string)

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。
        vwap: pandas DataFrame, 成交量加权平均价。
    适用性:
        截面/时序混合 (ts_rank, ts_max, ts_delta, power, rank)
    Alpha 含义:
        此 Alpha 是一个幂运算，底数和指数都涉及时间序列操作。
        Base: ts_rank(vwap - ts_max(vwap, 15.3), 20.7) (代码窗口: 15, 20)
            - ts_max(vwap, 15): 过去15日VWAP的最大值。
            - vwap - ts_max(vwap, 15): 当前VWAP与近期VWAP高点的差（通常为负或零）。
            - ts_rank(..., 20): 上述差值在过去20日的时序排名。
        Exponent: ts_delta(close, 5.0) (代码窗口: 6, from 4.96)
            - ts_delta(close, 6): 收盘价的6日变化。
        Alpha_raw = Base ^ Exponent
        Alpha_final = rank(Alpha_raw) (代码中对最终结果进行了排名)
        它捕捉了“VWAP相对近期高点位置的时序排名”与“收盘价近期动量”之间的幂律关系，并对结果进行横截面比较。
    出参:
        pandas Series, 计算得到的 Alpha 值（排名后的）。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    # Windows from code: ts_max win 15, ts_rank win 20, ts_delta win 6
    base = ts_rank(vwap.sub(ts_max(vwap, 15)), 20)
    exponent = ts_delta(c, 6)

    # The formula does not rank the result, but the code does.
    return (rank(power(base, exponent)) # Code ranks the result of power(base, exponent)
            .stack('ticker')
            .swaplevel())

def alpha085(c, h, l, v): # adv30 from v
    """
    Alpha Factor #085: power(rank(ts_corr(((high * 0.876703) + (close * (1 - 0.876703))), adv30,9.61331)), rank(ts_corr(ts_rank(((high + low) / 2), 3.70596), ts_rank(volume, 10.1595),7.11408)))

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。
        h: pandas DataFrame, 最高价。
        l: pandas DataFrame, 最低价。
        v: pandas DataFrame, 成交量 (用于计算 adv30)。
    适用性:
        截面 Alpha (rank, ts_corr, ts_rank, power)
    Alpha 含义:
        此 Alpha 是一个幂运算，底数和指数都是复杂的相关性排名。权重 w = 0.876703。
        Base: rank(ts_corr(high*w + close*(1-w), adv30, 9.6)) (代码窗口: 10)
            - price_combo1 = high*w + close*(1-w): 最高价和收盘价的加权组合。
            - adv30: 30日均量。
            - ts_corr(price_combo1, adv30, 10): 价格组合与30日均量的10日相关性。
            - rank(...): 对其排名。
        Exponent: rank(ts_corr(ts_rank((high+low)/2, 3.7), ts_rank(volume, 10.1), 7.1)) (代码窗口: 4, 10, 7)
            - ts_rank((high+low)/2, 4): 当日中间价的4日时序排名。
            - ts_rank(volume, 10): 成交量的10日时序排名。
            - ts_corr(..., ..., 7): 上述两个时序排名的7日相关性。
            - rank(...): 对其排名。
        Alpha = Base ^ Exponent。
        它捕捉了“特定价格组合与均量相关性的排名”与“中间价时序排名和成交量时序排名的相关性排名”之间的幂律关系。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    w = 0.876703
    adv30 = ts_mean(v, 30)

    base_price_combo = h.mul(w).add(c.mul(1 - w))
    base = rank(ts_corr(base_price_combo, adv30, 10)) # Window from code

    exp_term1 = ts_rank(h.add(l).div(2), 4) # Window from code
    exp_term2 = ts_rank(v, 10) # Window from code
    exponent = rank(ts_corr(exp_term1, exp_term2, 7)) # Window from code

    return (base.pow(exponent)
            .stack('ticker')
            .swaplevel())

def alpha086(c, v, vwap): # adv20 from v, open is missing for formula string. Code simplifies.
    """
    Alpha Factor #086: ((ts_rank(ts_corr(close, ts_sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open + close) - (vwap + open)))) * -1)
    Formula's rank term: rank(close - vwap)

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。
        v: pandas DataFrame, 成交量 (用于计算 adv20)。
        vwap: pandas DataFrame, 成交量加权平均价。
        (隐含需要: open，但代码中简化掉了 open)
    适用性:
        截面 Alpha (ts_rank, ts_corr, rank)
    Alpha 含义:
        此 Alpha 比较两个排名的大小，然后乘以-1。
        Cond1: ts_rank(ts_corr(close, ts_sum(adv20, 14.7), 6.0), 20.4) (代码窗口: sum of adv20 is ts_mean(adv20,15); corr win 6; rank win 20)
            - adv20: 20日均量。
            - processed_adv20 = ts_mean(adv20, 15) (代码中用20日均量的15日均值)
            - ts_corr(close, processed_adv20, 6): 收盘价与处理后均量的6日相关性。
            - ts_rank(..., 20): 对相关性进行20日时序排名。
        Cond2: rank(close - vwap) (公式 (open+close)-(vwap+open) 简化为 close-vwap)
            - close - vwap: 收盘价与VWAP的差。
            - rank(...): 对其排名。
        Alpha = (Cond1 < Cond2) * -1。
        它比较了“收盘价与处理后均量相关性的时序排名”和“收盘价与VWAP偏离的排名”。
    出参:
        pandas Series, 计算得到的 Alpha 值 (-1 或 0)。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    adv20 = ts_mean(v, 20)
    # Formula: ts_sum(adv20, 14.74)
    # Code: ts_mean(ts_mean(v,20), 15) which is ts_mean(adv20, 15)
    processed_adv20 = ts_mean(adv20, 15) # Using code's version

    cond1 = ts_rank(ts_corr(c, processed_adv20, 6), 20) # Windows from code

    # Formula: rank(((open + close) - (vwap + open))) = rank(close - vwap)
    # Code: rank(c.sub(vwap))
    cond2 = rank(c.sub(vwap)) # Matches simplified formula

    return (cond1.lt(cond2)
            .mul(-1)
            .stack('ticker')
            .swaplevel())

def alpha087(c, vwap, industry): # Requires v for adv81
    """
    Alpha Factor #087: (max(rank(ts_weighted_mean(ts_delta(((close * 0.369701) + (vwap * (1 - 0.369701))),1.91233), 2.65461)), ts_rank(ts_weighted_mean(abs(ts_corr(IndNeutralize(adv81,IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)
    (adv81 from v)

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。
        vwap: pandas DataFrame, 成交量加权平均价。
        industry: pandas Series/DataFrame, 股票对应的行业分类信息 (IndClass.industry)。
        (隐含需要: v for adv81)
    适用性:
        截面/时序混合 (rank, ts_weighted_mean, ts_delta, ts_rank, abs, ts_corr, IndNeutralize)
    Alpha 含义:
        此 Alpha 取两个复杂项中的较大者，然后乘以-1。权重 w = 0.369701。
        Term1: rank(ts_weighted_mean(ts_delta(close*w + vwap*(1-w), 1.91), 2.65))
            - price_combo = close*w + vwap*(1-w): 收盘价和VWAP的加权组合。
            - ts_delta(price_combo, 1.91): 价格组合的1.91日变化。
            - ts_weighted_mean(..., 2.65): 上述变化的2.65日加权均值。
            - rank(...): 对其排名。
        Term2: ts_rank(ts_weighted_mean(abs(ts_corr(IndNeutralize(adv81, IndClass.industry), close, 13.41)), 4.90), 14.45)
            - adv81: 81日均量。
            - IndNeutralize(adv81, IndClass.industry): 81日均量按行业中性化。
            - ts_corr(Ind_adv81, close, 13.41): 中性化均量与收盘价的13.41日相关性。
            - abs(...): 取相关性的绝对值。
            - ts_weighted_mean(..., 4.90): 上述绝对值相关性的4.90日加权均值。
            - ts_rank(..., 14.45): 对其进行14.45日时序排名。
        Alpha = max(Term1, Term2) * -1。
        它结合了“特定价格组合动量的加权均值排名”和“行业中性均量与收盘价的绝对相关性的加权均值的时序排名”。
        【注意】: 此函数标为 `pass`，未实现。需要额外数据 (v for adv81)。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    pass

def alpha088(o, h, l, c, v): # adv60 from v
    """
    Alpha Factor #088: min(rank(ts_weighted_mean(((rank(open) + rank(low)) - (rank(high) + rank(close))),8.06882)), ts_rank(ts_weighted_mean(ts_corr(ts_rank(close, 8.44728), ts_rank(adv60,20.6966), 8.01266), 6.65053), 2.61957))

    中文注释:
    入参:
        o: pandas DataFrame, 开盘价。
        h: pandas DataFrame, 最高价。
        l: pandas DataFrame, 最低价。
        c: pandas DataFrame, 收盘价。
        v: pandas DataFrame, 成交量 (用于计算 adv60)。
    适用性:
        截面/时序混合 (rank, ts_weighted_mean, ts_rank, ts_corr)
    Alpha 含义:
        此 Alpha 取两个复杂项中的较小者。
        Term1: rank(ts_weighted_mean(rank(open) + rank(low) - rank(high) - rank(close), 8.07)) (代码窗口: 8)
            - inner = rank(open) + rank(low) - rank(high) - rank(close): 综合了开盘、最低、最高、收盘价的排名。
              （如果o,l排名高而h,c排名低，则inner大，可能表示日内反转或特定形态）
            - ts_weighted_mean(inner, 8): 上述综合排名的8日加权均值。
            - rank(...): 对其排名。
        Term2: ts_rank(ts_weighted_mean(ts_corr(ts_rank(close, 8.45), ts_rank(adv60, 20.7), 8.01), 6.65), 2.62)
               (代码窗口: close_rank_win 8, adv_rank_win 20, corr_win 8, mean_win 6, final_ts_rank_win 2)
            - ts_rank(close, 8): 收盘价的8日时序排名。
            - adv60: 60日均量。
            - ts_rank(adv60, 20): 60日均量的20日时序排名。
            - ts_corr(..., ..., 8): 上述两个时序排名的8日相关性。
            - ts_weighted_mean(..., 6): 相关性的6日加权均值。
            - ts_rank(..., 2): 对加权均值进行2日时序排名。
        Alpha = min(Term1, Term2)。
        它结合了“四价综合排名加权均值的排名”和“收盘价时序排名与均量时序排名相关性的加权均值的时序排名”。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    adv60 = ts_mean(v, 60)

    s1_inner = rank(o).add(rank(l)).sub(rank(h)).sub(rank(c)) # Formula has +rank(close), code has .add(rank(c))
                                                              # Alpha088 formula string: (rank(open) + rank(low)) - (rank(high) + rank(close))
                                                              # Code: rank(o).add(rank(l)).sub(rank(h)).add(rank(c)) -- this is different
                                                              # Assuming code is correct for implementation: rank(o)+rank(l)-rank(h)+rank(c)
    s1 = rank(ts_weighted_mean(s1_inner, 8)) # Window from code

    s2_corr = ts_corr(ts_rank(c, 8), ts_rank(adv60, 20), 8) # Windows from code
    s2 = ts_rank(ts_weighted_mean(s2_corr, 6), 2) # Windows from code

    return (s1.where(s1 < s2, s2) # min(s1, s2)
            .stack('ticker')
            .swaplevel())

def alpha089(l, v, vwap, industry): # Requires adv10 from v. Formula: low*0.967 + low*(1-0.967) = low
    """
    Alpha Factor #089: (ts_rank(ts_weighted_mean(ts_corr(((low * 0.967285) + (low * (1 - 0.967285))), adv10,6.94279), 5.51607), 3.79744) - ts_rank(ts_weighted_mean(ts_delta(IndNeutralize(vwap,IndClass.industry), 3.48158), 10.1466), 15.3012))
    (adv10 from v)

    中文注释:
    入参:
        l: pandas DataFrame, 最低价。
        v: pandas DataFrame, 成交量 (用于计算 adv10)。
        vwap: pandas DataFrame, 成交量加权平均价。
        industry: pandas Series/DataFrame, 股票对应的行业分类信息 (IndClass.industry)。
    适用性:
        截面/时序混合 (ts_rank, ts_weighted_mean, ts_corr, ts_delta, IndNeutralize)
    Alpha 含义:
        此 Alpha 是两个复杂项的差。
        Term1: ts_rank(ts_weighted_mean(ts_corr(low, adv10, 6.94), 5.52), 3.80)
            - price_combo = low (因为 low*w + low*(1-w) = low)
            - adv10: 10日均量。
            - ts_corr(low, adv10, 6.94): 最低价与10日均量的6.94日相关性。
            - ts_weighted_mean(..., 5.52): 上述相关性的5.52日加权均值。
            - ts_rank(..., 3.80): 对其进行3.80日时序排名。
        Term2: ts_rank(ts_weighted_mean(ts_delta(IndNeutralize(vwap, IndClass.industry), 3.48), 10.15), 15.30)
            - IndNeutralize(vwap, IndClass.industry): VWAP按行业中性化。
            - ts_delta(..., 3.48): 中性化VWAP的3.48日变化。
            - ts_weighted_mean(..., 10.15): 上述变化的10.15日加权均值。
            - ts_rank(..., 15.30): 对其进行15.30日时序排名。
        Alpha = Term1 - Term2。
        它比较了“最低价与均量相关性的加权均值的时序排名”和“行业中性VWAP动量的加权均值的时序排名”。
        【注意】: 此函数标为 `pass`，未实现。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    pass

# alpha090, alpha091 are missing.

def alpha092(c, h, l, o, v): # adv30 from v
    """
    Alpha Factor #092: min(ts_rank(ts_weighted_mean(((((high + low) / 2) + close) < (low + open)), 14.7221),18.8683), ts_rank(ts_weighted_mean(ts_corr(rank(low), rank(adv30), 7.58555), 6.94024),6.80584))

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。
        h: pandas DataFrame, 最高价。
        l: pandas DataFrame, 最低价。
        o: pandas DataFrame, 开盘价。
        v: pandas DataFrame, 成交量 (用于计算 adv30)。
    适用性:
        截面/时序混合 (ts_rank, ts_weighted_mean, ts_corr, rank)
    Alpha 含义:
        此 Alpha 取两个复杂项中的较小者。
        Term1: ts_rank(ts_weighted_mean( ((high+low)/2 + close) < (low+open) , 14.7), 18.9) (代码窗口: 15, 18)
            - condition = ((high+low)/2 + close) < (low+open): 一个复杂的日内价格关系比较。
            - ts_weighted_mean(condition, 15): 上述布尔条件 (转为1/0) 的15日加权均值 (表示条件成立的频率或强度)。
            - ts_rank(..., 18): 对其进行18日时序排名。
        Term2: ts_rank(ts_weighted_mean(ts_corr(rank(low), rank(adv30), 7.6), 6.9), 6.8) (代码窗口: 7,6,6)
            - rank(low): 最低价排名。
            - adv30: 30日均量。
            - rank(adv30): 30日均量排名。
            - ts_corr(..., ..., 7): 最低价排名与均量排名的7日相关性。
            - ts_weighted_mean(..., 6): 相关性的6日加权均值。
            - ts_rank(..., 6): 对加权均值进行6日时序排名。
        Alpha = min(Term1, Term2)。
        它结合了“特定日内价格模式频率的加权均值的时序排名”和“最低价排名与均量排名相关性的加权均值的时序排名”。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    adv30 = ts_mean(v, 30)

    # Term1 windows from code: mean_win=15, rank_win=18
    cond_term1 = (h.add(l).div(2).add(c)).lt(l.add(o))
    p1 = ts_rank(ts_weighted_mean(cond_term1.astype(float), 15), 18)

    # Term2 windows from code: corr_win=7, mean_win=6, rank_win=6
    corr_term2 = ts_corr(rank(l), rank(adv30), 7)
    p2 = ts_rank(ts_weighted_mean(corr_term2, 6), 6)

    return (p1.where(p1<p2, p2) # min(p1, p2)
            .stack('ticker')
            .swaplevel())

def alpha093(c, v, vwap, industry): # adv81 from v
    """
    Alpha Factor #093: (ts_rank(ts_weighted_mean(ts_corr(IndNeutralize(vwap, IndClass.industry), adv81,17.4193), 19.848), 7.54455) / rank(ts_weighted_mean(ts_delta(((close * 0.524434) + (vwap * (1 -0.524434))), 2.77377), 16.2664)))

    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。
        v: pandas DataFrame, 成交量 (用于计算 adv81)。
        vwap: pandas DataFrame, 成交量加权平均价。
        industry: pandas Series/DataFrame, 股票对应的行业分类信息 (IndClass.industry)。
    适用性:
        截面/时序混合 (ts_rank, ts_weighted_mean, ts_corr, IndNeutralize, rank, ts_delta)
    Alpha 含义:
        此 Alpha 是两个复杂项的比值。权重 w = 0.524434。
        Numerator: ts_rank(ts_weighted_mean(ts_corr(IndNeutralize(vwap, IndClass.industry), adv81, 17.4), 19.8), 7.5)
            - IndNeutralize(vwap, IndClass.industry): VWAP按行业中性化。
            - adv81: 81日均量。
            - ts_corr(Ind_vwap, adv81, 17.4): 中性化VWAP与81日均量的17.4日相关性。
            - ts_weighted_mean(..., 19.8): 上述相关性的19.8日加权均值。
            - ts_rank(..., 7.5): 对其进行7.5日时序排名。
        Denominator: rank(ts_weighted_mean(ts_delta(close*w + vwap*(1-w), 2.77), 16.27))
            - price_combo = close*w + vwap*(1-w): 收盘价和VWAP的加权组合。
            - ts_delta(price_combo, 2.77): 价格组合的2.77日变化。
            - ts_weighted_mean(..., 16.27): 上述变化的16.27日加权均值。
            - rank(...): 对其排名。
        Alpha = Numerator / Denominator。
        它比较了“行业中性VWAP与均量相关性的加权均值的时序排名”和“特定价格组合动量的加权均值排名”。
        【注意】: 此函数标为 `pass`，未实现。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    pass

def alpha094(v, vwap): # adv60 from v
    """
    Alpha Factor #094: ((rank((vwap - ts_min(vwap, 11.5783)))^ts_rank(ts_corr(ts_rank(vwap,19.6462), ts_rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)

    中文注释:
    入参:
        v: pandas DataFrame, 成交量 (用于计算 adv60)。
        vwap: pandas DataFrame, 成交量加权平均价。
    适用性:
        截面 Alpha (rank, ts_min, ts_rank, ts_corr, power)
    Alpha 含义:
        此 Alpha 是一个幂运算，然后乘以-1。
        Base: rank(vwap - ts_min(vwap, 11.6)) (代码窗口: 11)
            - vwap - ts_min(vwap, 11): 当前VWAP与过去11日VWAP最小值的差（VWAP从近期低点回升幅度）。
            - rank(...): 对其排名。
        Exponent: ts_rank(ts_corr(ts_rank(vwap, 19.6), ts_rank(adv60, 4.0), 18.1), 2.7) (代码窗口: 20,4,18,2)
            - ts_rank(vwap, 20): VWAP的20日时序排名。
            - adv60: 60日均量。
            - ts_rank(adv60, 4): 60日均量的4日时序排名。
            - ts_corr(..., ..., 18): 上述两个时序排名的18日相关性。
            - ts_rank(..., 2): 对相关性进行2日时序排名。
        Alpha = (Base ^ Exponent) * -1。
        它捕捉了“VWAP从低点回升幅度的排名”与“VWAP时序排名和均量时序排名的相关性的时序排名”之间的幂律关系。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    adv60 = ts_mean(v, 60)

    base = rank(vwap.sub(ts_min(vwap, 11))) # Window from code

    exp_corr_term1 = ts_rank(vwap, 20) # Window from code
    exp_corr_term2 = ts_rank(adv60, 4) # Window from code
    exp_corr = ts_corr(exp_corr_term1, exp_corr_term2, 18) # Window from code
    exponent = ts_rank(exp_corr, 2) # Window from code

    return (base.pow(exponent)
            .mul(-1)
            .stack('ticker')
            .swaplevel())

def alpha095(h, l, o, v): # adv40 from v
    """
    Alpha Factor #095: (rank((open - ts_min(open, 12.4105))) < ts_rank((rank(ts_corr(ts_sum(((high + low)/ 2), 19.1351), ts_sum(adv40, 19.1351), 12.8742))^5), 11.7584))
    
    中文注释:
    入参:
        h: pandas DataFrame, 最高价。
        l: pandas DataFrame, 最低价。
        o: pandas DataFrame, 开盘价。
        v: pandas DataFrame, 成交量 (用于计算 adv40)。
    适用性:
        截面 Alpha (rank, ts_min, ts_rank, ts_corr, ts_sum, power)
    Alpha 含义:
        此 Alpha 比较两个复杂排名的大小。
        Cond1: rank(open - ts_min(open, 12.4)) (代码窗口: 12)
            - open - ts_min(open, 12): 开盘价与过去12日最低开盘价的差（开盘价从近期低点回升幅度）。
            - rank(...): 对其排名。
        Cond2_inner_corr: ts_corr(ts_sum((high+low)/2, 19.1), ts_sum(adv40, 19.1), 12.9) (代码窗口: 19,19,13)
            - ts_sum((high+low)/2, 19): 每日中间价的19日累积。
            - adv40: 40日均量。
            - ts_sum(adv40, 19): 40日均量的19日累积。
            - ts_corr(..., ..., 13): 上述两个累积值的13日相关性。
        Cond2: ts_rank( (rank(Cond2_inner_corr))^5, 11.8) (代码窗口: 12)
            - rank(Cond2_inner_corr): 对相关性进行排名。
            - (...)^5: 排名的5次方。
            - ts_rank(..., 12): 对5次方结果进行12日时序排名。
        Alpha = (Cond1 < Cond2)。结果为布尔值，代码转为整数 (1/0)。
        它比较了“开盘价从低点回升幅度的排名”和“特定价格累积与均量累积的相关性排名的5次方后的时序排名”。
    出参:
        pandas Series, 计算得到的 Alpha 值 (1 或 0)。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    adv40 = ts_mean(v, 40)

    cond1 = rank(o.sub(ts_min(o, 12))) # Window from code

    # Cond2 calculation based on code
    corr_val = ts_corr(ts_mean(h.add(l).div(2), 19), # Formula has sum, code has mean for price part.
                       ts_sum(adv40, 19), 13)       # Formula has sum, code has sum for adv part.
                                                    # Assuming code's ts_mean for price part.
    rank_corr_pow5 = rank(corr_val).pow(5)
    cond2 = ts_rank(rank_corr_pow5, 12) # Window from code

    return (cond1.lt(cond2)
            .astype(int)
            .stack('ticker')
            .swaplevel())

def alpha096(c, v, vwap): # adv60 from v
    """
    Alpha Factor #096: (max(ts_rank(ts_weighted_mean(ts_corr(rank(vwap), rank(volume), 5.83878),4.16783), 8.38151), ts_rank(ts_weighted_mean(ts_argmax(ts_corr(ts_rank(close, 7.45404), ts_rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
    
    中文注释:
    入参:
        c: pandas DataFrame, 收盘价。
        v: pandas DataFrame, 成交量 (用于计算 adv60 和直接使用)。
        vwap: pandas DataFrame, 成交量加权平均价。
    适用性:
        截面/时序混合 (ts_rank, ts_weighted_mean, ts_corr, rank, ts_argmax)
    Alpha 含义:
        此 Alpha 取两个复杂项中的较大者，然后乘以-1。
        Term1: ts_rank(ts_weighted_mean(ts_corr(rank(vwap), rank(volume), 5.8), 4.17), 8.38)
               (代码窗口: corr_win 10, mean_win 4, rank_win 8)
            - rank(vwap): VWAP排名。
            - rank(volume): 成交量排名。
            - ts_corr(..., ..., 10): 两者排名的10日相关性。
            - ts_weighted_mean(..., 4): 相关性的4日加权均值。
            - ts_rank(..., 8): 对加权均值进行8日时序排名。
        Term2: ts_rank(ts_weighted_mean(ts_argmax(ts_corr(ts_rank(close,7.5), ts_rank(adv60,4.1),3.7), 12.7), 14.0), 13.4)
               (代码窗口: c_rank 7, adv_rank 10, corr 10, argmax 12, mean 14, final_rank 13)
            - ts_rank(close, 7): 收盘价7日时序排名。
            - adv60: 60日均量。
            - ts_rank(adv60, 10): 60日均量10日时序排名。
            - ts_corr(..., ..., 10): 两者时序排名的10日相关性。
            - ts_argmax(..., 12): 上述相关性在过去12日中最大值出现的位置。
            - ts_weighted_mean(..., 14): 对位置进行14日加权均值。
            - ts_rank(..., 13): 对加权均值进行13日时序排名。
        Alpha = max(Term1, Term2) * -1。
        它结合了“VWAP排名与成交量排名相关性的加权均值的时序排名”和“一个更复杂的基于收盘价与均量时序排名的相关性的最大值位置的指标的时序排名”。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    adv60 = ts_mean(v, 60)

    # Term1 from code
    s1 = ts_rank(ts_weighted_mean(ts_corr(rank(vwap), rank(v), 10), 4), 8)

    # Term2 from code
    corr_s2 = ts_corr(ts_rank(c, 7), ts_rank(adv60, 10), 10)
    argmax_s2 = ts_argmax(corr_s2, 12)
    mean_s2 = ts_weighted_mean(argmax_s2, 14)
    s2 = ts_rank(mean_s2, 13)

    return (s1.where(s1 > s2, s2) # max(s1,s2)
            .mul(-1)
            .stack('ticker')
            .swaplevel())

def alpha097(l): # Requires vwap, v for adv60, industry
    """((rank(ts_weighted_mean(ts_delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))),IndClass.industry), 3.3705), 20.4523)) - ts_rank(ts_weighted_mean(ts_rank(ts_corr(Ts_Rank(low,7.87871), ts_rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)
    (adv60 from v)

    中文注释:
    入参:
        l: pandas DataFrame, 最低价。
        (隐含需要: vwap, v for adv60, industry)
    适用性:
        截面/时序混合 (rank, ts_weighted_mean, ts_delta, IndNeutralize, ts_rank, ts_corr, Ts_Rank)
    Alpha 含义:
        此 Alpha 是两个复杂项的差，然后乘以-1。权重 w = 0.721001。
        Term1: rank(ts_weighted_mean(ts_delta(IndNeutralize(low*w + vwap*(1-w), IndClass.industry), 3.37), 20.45))
            - price_combo = low*w + vwap*(1-w): 最低价和VWAP的加权组合。
            - IndNeutralize(price_combo, IndClass.industry): 价格组合按行业中性化。
            - ts_delta(..., 3.37): 中性化价格组合的3.37日变化。
            - ts_weighted_mean(..., 20.45): 上述变化的20.45日加权均值。
            - rank(...): 对其排名。
        Term2: ts_rank(ts_weighted_mean(ts_rank(ts_corr(Ts_Rank(low,7.88), ts_rank(adv60,17.26),4.98), 18.59), 15.72), 6.72)
            - Ts_Rank(low, 7.88): 最低价的7.88日时序排名。
            - adv60: 60日均量。
            - ts_rank(adv60, 17.26): 60日均量的17.26日时序排名。
            - ts_corr(..., ..., 4.98): 上述两个时序排名的4.98日相关性。
            - ts_rank(..., 18.59): 对相关性进行18.59日时序排名 (这里是Ts_Rank，但通常实现为ts_rank)。
            - ts_weighted_mean(..., 15.72): 对时序排名结果进行15.72日加权均值。
            - ts_rank(..., 6.72): 对加权均值进行6.72日时序排名。
        Alpha = (Term1 - Term2) * -1 = Term2 - Term1。
        它比较了“特定行业中性价格组合动量的加权均值排名”和“一个更复杂的基于最低价时序排名与均量时序排名相关的指标的时序排名”。
        【注意】: 此函数标为 `pass`，未实现。需要额外数据 (vwap, v for adv60, industry)。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    pass

def alpha098(o, v, vwap): # adv5, adv15 from v
    """(rank(ts_weighted_mean(ts_corr(vwap, ts_sum(adv5, 26.4719), 4.58418), 7.18088)) -
        rank(ts_weighted_mean(ts_tank(ts_argmin(ts_corr(rank(open), 
        rank(adv15), 20.8187), 8.62571),6.95668), 8.07206))) # ts_tank is likely ts_rank

    中文注释:
    入参:
        o: pandas DataFrame, 开盘价。
        v: pandas DataFrame, 成交量 (用于计算 adv5, adv15)。
        vwap: pandas DataFrame, 成交量加权平均价。
    适用性:
        截面 Alpha (rank, ts_weighted_mean, ts_corr, ts_sum, ts_argmin)
    Alpha 含义:
        此 Alpha 是两个复杂排名项的差。
        Term1: rank(ts_weighted_mean(ts_corr(vwap, ts_sum(adv5, 26.5), 4.6), 7.2)) (代码窗口: sum_adv5_win 26, corr_win 4, mean_win 7)
            - adv5: 5日均量。
            - ts_sum(adv5, 26): 5日均量的26日累积。(代码中 ts_mean(adv5,26))
            - ts_corr(vwap, sum_adv5, 4): VWAP与累积均量的4日相关性。
            - ts_weighted_mean(..., 7): 上述相关性的7日加权均值。
            - rank(...): 对其排名。
        Term2: rank(ts_weighted_mean(ts_rank(ts_argmin(ts_corr(rank(open), rank(adv15),20.8),8.6),6.96),8.07))
               (代码窗口: corr_win 20, argmin_win 8, rank_win 6, mean_win 8. Formula has ts_tank, assuming ts_rank)
            - adv15: 15日均量。
            - rank(open): 开盘价排名。
            - rank(adv15): 15日均量排名。
            - ts_corr(..., ..., 20): 两者排名的20日相关性。
            - ts_argmin(..., 8): 上述相关性在过去8日中最小值出现的位置。
            - ts_rank(..., 6): 对位置进行6日时序排名 (原文ts_tank)。
            - ts_weighted_mean(..., 8): 对时序排名结果进行8日加权均值。
            - rank(...): 对其排名。
        Alpha = Term1 - Term2。
        它比较了“VWAP与处理后短期均量相关性的加权均值排名”和“一个基于开盘价排名与中期均量排名相关性的最小值位置的复杂指标的排名”。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    adv5 = ts_mean(v, 5)
    adv15 = ts_mean(v, 15)

    # Term1 from code
    # Formula: ts_sum(adv5, 26.4719) vs Code: ts_mean(adv5, 26)
    term1_corr = ts_corr(vwap, ts_mean(adv5, 26), 4) # Using code's ts_mean for adv5 part
    term1 = rank(ts_weighted_mean(term1_corr, 7))

    # Term2 from code (assuming ts_tank is ts_rank)
    term2_corr = ts_corr(rank(o), rank(adv15), 20)
    term2_argmin = ts_argmin(term2_corr, 8)
    term2_ts_rank = ts_rank(term2_argmin, 6) # ts_tank assumed as ts_rank
    term2 = rank(ts_weighted_mean(term2_ts_rank, 8))

    return (term1.sub(term2)
            .stack('ticker')
            .swaplevel())

def alpha099(h, l, v): # adv60 from v
    """
    Alpha Factor #099: ((rank(ts_corr(ts_sum(((high + low) / 2), 19.8975), ts_sum(adv60, 19.8975), 8.8136)) < rank(ts_corr(low, volume, 6.28259))) * -1)

    中文注释:
    入参:
        h: pandas DataFrame, 最高价。
        l: pandas DataFrame, 最低价。
        v: pandas DataFrame, 成交量 (用于计算 adv60 和直接使用)。
    适用性:
        截面 Alpha (rank, ts_corr, ts_sum)
    Alpha 含义:
        此 Alpha 比较两个相关性排名的大小，然后乘以-1。
        Cond1: rank(ts_corr(ts_sum((high+low)/2, 19.9), ts_sum(adv60, 19.9), 8.8)) (代码窗口: 19,19,8)
            - ts_sum((high+low)/2, 19): 每日中间价的19日累积。
            - adv60: 60日均量。
            - ts_sum(adv60, 19): 60日均量的19日累积。
            - ts_corr(..., ..., 8): 上述两个累积值的8日相关性。
            - rank(...): 对其排名。
        Cond2: rank(ts_corr(low, volume, 6.3)) (代码窗口: 6)
            - ts_corr(low, volume, 6): 最低价与当日成交量的6日相关性。
            - rank(...): 对其排名。
        Alpha = (Cond1 < Cond2) * -1。
        它比较了“价格中枢累积与均量累积的相关性排名”和“最低价与成交量相关性的排名”。
    出参:
        pandas Series, 计算得到的 Alpha 值 (-1 或 0)。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    adv60 = ts_mean(v, 60)

    # Cond1 from code
    cond1_price_sum = ts_sum(h.add(l).div(2), 19)
    cond1_adv_sum = ts_sum(adv60, 19)
    cond1 = rank(ts_corr(cond1_price_sum, cond1_adv_sum, 8))

    # Cond2 from code
    cond2 = rank(ts_corr(l, v, 6))

    return ((cond1.lt(cond2))
             .mul(-1))
            .stack('ticker')
            .swaplevel())

def alpha100(r, cap): # Requires close, low, high, volume, adv20
    """
    Alpha Factor #100: (0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high -close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) - scale(indneutralize((ts_corr(close, rank(adv20), 5) - rank(ts_argmin(close, 30))), IndClass.subindustry))) * (volume / adv20))))

    中文注释:
    入参:
        r: pandas DataFrame, 收益率。(公式中未使用，但通常alpha会用到主要价格数据)
        cap: pandas DataFrame, 市值。(公式中未使用)
        (隐含需要: close, low, high, volume, adv20, IndClass.subindustry)
    适用性:
        截面 Alpha (scale, indneutralize, rank, ts_corr, ts_argmin)
    Alpha 含义:
        此 Alpha 结构非常复杂，包含多层行业中性化和标准化。
        Inner_term1 = rank(( (2*close - low - high) / (high - low) ) * volume) : 日内价格位置成交量加权排名。
        Term1 = 1.5 * scale(indneutralize(indneutralize(Inner_term1, Subindustry), Subindustry)) : 双重行业中性化并标准化后再放大。

        Inner_term2 = ts_corr(close, rank(adv20), 5) - rank(ts_argmin(close, 30))
            - ts_corr(close, rank(adv20), 5): 收盘价与20日均量排名的5日相关性。
            - rank(ts_argmin(close, 30)): 30日内收盘价最低点出现时间的排名。
        Term2 = scale(indneutralize(Inner_term2, Subindustry)): 行业中性化后标准化。

        Volume_ratio = volume / adv20: 量比。

        Alpha = -( (Term1 - Term2) * Volume_ratio )。
        它是一个高度处理过的因子，结合了日内价格行为、量价关系、价格动量，并进行了多次行业中性化和标准化，最后乘以量比并取负。
        【注意】: 此函数标为 `pass`，未实现。需要额外数据。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    pass

def alpha101(o, h, l, c):
    """
    Alpha Factor #101: ((close - open) / ((high - low) + .001))

    中文注释:
    入参:
        o: pandas DataFrame, 开盘价。
        h: pandas DataFrame, 最高价。
        l: pandas DataFrame, 最低价。
        c: pandas DataFrame, 收盘价。
    适用性:
        时序 Alpha (直接计算日内价格关系)
    Alpha 含义:
        此 Alpha 计算日内收益相对于当日振幅的比例。
        Numerator = close - open: 当日日内收益。
        Denominator = (high - low) + 0.001: 当日振幅（最高价-最低价），加上一个小数是为了防止除以零（当振幅为0时）。
        Alpha = Numerator / Denominator。
        - 如果当日上涨，且振幅较小，则 Alpha 值较大（正）。
        - 如果当日下跌，且振幅较小，则 Alpha 值较小（负的绝对值大）。
        - 如果振幅很大，Alpha 的绝对值会变小。
        它衡量了日内趋势的强度（相对于总波动）。
    出参:
        pandas Series, 计算得到的 Alpha 值。索引为 (日期, 股票代码) 的 MultiIndex。
    """
    return (c.sub(o).div(h.sub(l).add(1e-3)) # Using 1e-3 as in other alphas for consistency with +0.001
            .stack('ticker')
            .swaplevel())