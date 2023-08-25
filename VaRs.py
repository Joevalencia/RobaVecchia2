import pandas as pd
import numpy as np
import scipy.stats as sps
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from seaborn import set_style

set_style('darkgrid')


def riesgos(firm: str, start: str, end: str,
            investment: float, parametric: bool = True) -> (str, float, bool):

    import pandas_datareader.data as web
    import scipy.stats as sps

    firm = firm
    c1 = start
    c2 = end
    investment = investment
    f3 = web.DataReader(firm, 'yahoo', c1, c2)['Adj Close']
    f3_log = np.log(f3 / f3.shift(1)).dropna()
    w1, w2, w3, w4, w5, w6 = sps.describe(f3_log)
    print('Estadisticos de la distribucion logaritmica')
    print('Media: ',round(w3, 4))
    print('Varianza: ', round(w4, 4))
    print('Asimetria: ', round(w5, 4))
    print('Curtosis: ', round(w6, 4))
    print('--------------------------------------------')
    x = np.linspace(np.min(f3_log), np.max(f3_log), 1000)
    if parametric:
        mu_norm, sig_norm = sps.norm.fit(f3_log)
        zeta = sps.norm.ppf(.01)
        pdf_nor = sps.norm.pdf(x, mu_norm, sig_norm)
        nu, mu_t, sig_t = sps.t.fit(f3_log)
        pdf_t = sps.t.pdf(x, nu, mu_t, sig_t)

        h = 1  # days
        alpha = .01  # significance level
        StudenthVaR = h * mu_norm - (h * (nu - 2) / nu) ** 0.5 * sps.t.ppf(1 - alpha, nu) * sig_norm
        NormalhVaR = h * mu_norm - sps.norm.ppf(1 - alpha) * sig_norm * (h ** 0.5)

        lev = 100 * (1 - alpha)
        print('Percentage Values of VaRs')
        print("%g%% %g-day Student t VaR = %.3f%%" % (lev, h, StudenthVaR * 100))
        print("%g%% %g-day Normal VaR    = %.3f%%" % (lev, h, NormalhVaR * 100))

        print('------------------------------------------------------------------------')
        print('------------------------------------------------------------------------')
        print(f'Monetary amount of Loss according to Normal VaR given {investment} bucks:\n ')
        sa = ((1 - np.exp(NormalhVaR)) * investment)
        print(round(sa, 2))
        print('------------------------------------------------------------------------')
        print(f'Monetary amount of Loss according to t-Student VaR given {investment} bucks:\n ')
        print(round((1 - np.exp(StudenthVaR)) * investment, 2))
        print('------------------------------------------------------------------------')
        print('------------------------------------------------------------------------')
        z = (1 / alpha * 1 / np.sqrt(2 * np.pi) * np.exp(-.5 * zeta ** 2))  # Normale
        CVaR_normale = mu_norm - sig_norm * z
        print("%g%% %g-day Normal CVaR  = %.3f%%" % (lev, h, CVaR_normale * 100))
        Cvar_t = mu_t + sig_t * sps.t.ppf(.01, nu)
        print("%g%% %g-day t-Student CVaR  = %.3f%%" % (lev, h, Cvar_t * 100))
        CVaR_norm = ((1 - np.exp(CVaR_normale)) * investment)
        print('------------------------------------------------------------------------')
        print(f'Monetary amount of Loss according to Normal CVaR given {investment}:\n ')
        print(CVaR_norm.round(2))
        print(f'Monetary amount of Loss according to t-Stud CVaR given {investment}:\n ')
        valor = (1 - np.exp(Cvar_t)) * investment
        print(round(valor, 2))

        # Graphs
        plt.rcParams['figure.figsize'] = [17, 5]
        p0 = plt.subplot(3, 1, 1)
        p0.plot(f3_log)
        plt.ylabel('En logaritmos', fontweight='bold')
        p1 = plt.subplot(3, 1, 2)
        p1.plot(f3)
        plt.ylabel('Asset', fontweight='bold')
        p2 = plt.subplot(3, 1, 3)
        p2.hist(f3_log, edgecolor='black', alpha=.5, color=(.8, .8, .8), bins='fd', density=True)
        p2.plot(x, pdf_nor, label='Normal', color='darkblue')
        p2.plot(x, pdf_t, label='t-Student', color='darkgreen')
        plt.ylabel('Log Returns and VaR 1%', fontweight='bold')
        plt.axvline(StudenthVaR, label='VaR 1%', color='red', linewidth=2)
        plt.legend()
        plt.show()

    else:
        order = f3_log.sort_values(ascending=True)
        valu = np.percentile(order, 1)
        print(f'Percentile value {round(valu * 100, 3)}')
        var01 = ((1 - np.exp(valu)) * investment)
        print(f'Historical Value-at-Risk given {investment} bucks:\n'
              f'The VaR for this asset in monetary terms is {round(var01, 2)}')
        print('-----------------------------------------------------------')
        cvar = (order[order <= valu]).mean()
        print(f'Historical Conditional Value-at-Risk given {investment} bucks:\n'
              f'Percentile value {round(cvar * 100, 2)}%')
        print('-----------------------------------------------------------')
        print('Monetary amount of Loss using Historical CVaR for this asset is:')
        money = (1 - np.exp(cvar)) * investment
        print(money.round(2))

        # PLOTTING
        plt.rcParams['figure.figsize'] = [17, 5]
        p0 = plt.subplot(3, 1, 1)
        plt.plot(f3)
        plt.ylabel('Retornos', fontweight='bold')
        p1 = plt.subplot(3, 1, 2)
        plt.ylabel('Asset', fontweight='bold')
        plt.plot(f3_log)
        p2 = plt.subplot(3, 1, 3)
        plt.ylabel('VaR 1%', fontweight='bold')
        plt.hist(f3_log, edgecolor='black', alpha=.5, color='greenyellow', bins='fd', density=True)
        plt.axvline(valu, label='VaR 1%', color='red', linewidth=2)
        plt.axvline(cvar, label='CVaR 1%', color='darkslategrey', linewidth=2)
        plt.legend()
        plt.show()



def report(firm: str, start: str, end: str, investment: float) -> (str, float):
    import pandas_datareader.data as web
    import scipy.stats as sps

    firm = firm
    c1 = start
    c2 = end
    investment = investment
    f3 = web.DataReader(firm, 'yahoo', c1, c2)['Adj Close']
    f3_log = np.log(f3 / f3.shift(1)).dropna()
    x = np.linspace(np.min(f3_log), np.max(f3_log), 1000)
    fit_norm = sps.norm.fit(f3_log)
    pdf_norm = sps.norm.pdf(x, fit_norm[0], fit_norm[1])
    fit_t = sps.t.fit(f3_log)
    pdf_t = sps.t.pdf(x, fit_t[0], fit_t[1], fit_t[2])

    order = f3_log.sort_values(ascending=True)
    valu = np.percentile(order, 1)
    print(f'Percentile value {round(valu * 100, 3)}%')
    var01 = ((1 - np.exp(valu)) * investment)
    print(f'Historical Value-at-Risk given {investment} bucks:\n'
          f'The VaR for this asset in monetary terms is {round(var01, 2)}')
    print('-----------------------------------------------------------')
    cvar = (order[order <= valu]).mean()
    print(f'Historical Conditional Value-at-Risk given {investment} bucks:\n'
          f'Percentile value {round(cvar * 100, 2)}%')
    print('-----------------------------------------------------------')
    print('Monetary amount of Loss using Historical CVaR for this asset is:')
    money = (1 - np.exp(cvar)) * investment
    print(money.round(2))

    # Plotting

    plt.rcParams['figure.figsize'] = [15, 4]
    p1 = plt.subplot(3, 1, 1)
    plt.ylabel('Asset', fontweight='bold')
    plt.plot(f3_log)
    # plt.xlabel('Log Returns')
    p2 = plt.subplot(3, 1, 2)
    plt.ylabel('Serie Temporal', fontweight='bold')
    plt.plot(f3)
    p3 = plt.subplot(3, 1, 3)
    plt.ylabel('Historical VaR 1%', fontweight='bold')
    plt.hist(f3_log, edgecolor='black', alpha=.5, color='greenyellow', bins='fd', density=True)
    plt.axvline(valu, label='VaR 1%', color='red', linewidth=2)
    plt.axvline(cvar, label='CVaR 1%', color='darkslategrey', linewidth=2)
    plt.legend()
    plt.show()


activo = 'TXN'
inicio = '2018-04-30'
fin = '2020-04-30'

#riesgos(firm=activo, start=inicio, end=fin, investment=1000, parametric=True)

# texas = web.DataReader('amzn', 'yahoo', '2018-04-30', '2020-04-30')['Adj Close']
# print(texas.head())


class Value_at_Risk:

    def VaR(self, vector, investement: float, a: float, parametric: bool = True) -> float:
        """
        This method aims at calculating and plotting either Parametric VaR and Non-parametric VaR
        The distribution used are Normal and T-Students (given logreturns)
        """
        if parametric:

            log_re = np.log(vector / vector.shift(1)).dropna()
            x = np.linspace(np.min(log_re), np.max(log_re), 1000)
            mu_norm, sig_norm = sps.norm.fit(log_re)
            zeta = sps.norm.ppf(a)
            pdf_nor = sps.norm.pdf(x, mu_norm, sig_norm)
            nu, mu_t, sig_t = sps.t.fit(log_re)
            pdf_t = sps.t.pdf(x, nu, mu_t, sig_t)

            h = 1  # days
            alpha = a  # significance level
            StudenthVaR = h * mu_norm - (h * (nu - 2) / nu) ** 0.5 * sps.t.ppf(1 - alpha, nu) * sig_norm
            NormalhVaR = h * mu_norm - sps.norm.ppf(1 - alpha) * sig_norm * (h ** 0.5)

            lev = 100 * (1 - alpha)
            print('Percentage Values of VaRs')
            print("%g%% %g-day Student t VaR = %.3f%%" % (lev, h, StudenthVaR * 100))
            print("%g%% %g-day Normal VaR    = %.3f%%" % (lev, h, NormalhVaR * 100))

            print('------------------------------------------------------------------------')
            print('------------------------------------------------------------------------')
            print(f'Monetary amount of Loss according to Normal VaR given {investement} bucks:\n ')
            sa = ((1 - np.exp(NormalhVaR)) * investement)
            print(round(sa, 2))
            print('------------------------------------------------------------------------')
            print(f'Monetary amount of Loss according to t-Student VaR given {investement} bucks:\n ')
            print(round((1 - np.exp(StudenthVaR)) * investement, 2))
            print('------------------------------------------------------------------------')
            print('------------------------------------------------------------------------')
            z = (1 / alpha * 1 / np.sqrt(2 * np.pi) * np.exp(-.5 * zeta ** 2))  # Normale
            CVaR_normale = mu_norm - sig_norm * z
            print("%g%% %g-day Normal CVaR  = %.3f%%" % (lev, h, CVaR_normale * 100))
            Cvar_t = mu_t + sig_t * sps.t.ppf(a, nu)
            print("%g%% %g-day t-Student CVaR  = %.3f%%" % (lev, h, Cvar_t * 100))
            CVaR_norm = ((1 - np.exp(CVaR_normale)) * investement)
            print('------------------------------------------------------------------------')
            print(f'Monetary amount of Loss according to Normal CVaR given {investement}:\n ')
            print(CVaR_norm.round(2))
            print(f'Monetary amount of Loss according to t-Stud CVaR given {investement}:\n ')
            valor = (1 - np.exp(Cvar_t)) * investement
            print(round(valor, 2))

            # Graphs
            plt.rcParams['figure.figsize'] = [13, 5]
            p1 = plt.subplot(1, 2, 1)
            p1.plot(vector)
            plt.title('Asset', fontweight='bold')
            plt.xlabel('Historical Serie')
            p2 = plt.subplot(1, 2, 2)
            p2.hist(log_re, edgecolor='black', alpha=.5, color=(.8, .8, .8), bins='fd', density=True)
            p2.plot(x, pdf_nor, label='Normal', color='darkblue')
            p2.plot(x, pdf_t, label='t-Student', color='darkgreen')
            plt.title(f'Log Returns and VaR {a * 100}%', fontweight='bold')
            plt.xlabel('Log Returns')
            plt.axvline(StudenthVaR, label=f'VaR {a * 100}%', color='red', linewidth=2)
            plt.legend()
            plt.show()

        else:

            log_re = np.log(vector / vector.shift(1)).dropna()
            order = log_re.sort_values(ascending=True)
            valu = np.percentile(order, a * 100)
            print(f'Percentile value {round(valu * 100, 3)}')
            var01 = ((1 - np.exp(valu)) * investement)
            print(f'Historical Value-at-Risk given {investement} bucks:\n'
                  f'The VaR for this asset in monetary terms is {round(var01, 2)}')
            print('-----------------------------------------------------------')
            cvar = (order[order <= valu]).mean()
            print(f'Historical Conditional Value-at-Risk given {investement} bucks:\n'
                  f'Percentile value {round(cvar * 100, 2)}%')
            print('-----------------------------------------------------------')
            print('Monetary amount of Loss using Historical CVaR for this asset is:')
            money = (1 - np.exp(cvar)) * investement
            print(money.round(2))

            # PLOTTING
            plt.rcParams['figure.figsize'] = [13, 5]
            p1 = plt.subplot(1, 2, 1)
            plt.title('Asset', fontweight='bold')
            plt.plot(vector)
            plt.xlabel('Log Returns')
            p2 = plt.subplot(1, 2, 2)
            plt.title(f'Historical VaR {a * 100}%', fontweight='bold')
            plt.hist(log_re, edgecolor='black', alpha=.5, color='greenyellow', bins='fd', density=True)
            plt.axvline(valu, label=f'VaR {a * 100}%', color='red', linewidth=2)
            plt.axvline(cvar, label=f'CVaR {a * 100}%', color='darkslategrey', linewidth=2)
            plt.legend()
            plt.show()

# mirror = Value_at_Risk()
# print(mirror.VaR(texas, 10000, a=0.01, parametric=True))
