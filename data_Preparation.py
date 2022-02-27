# %%
import json
import time

import pandas as pd
import yfinance as yf
from datetime import timedelta, date
import numpy as np

# finnhub_client = finnhub.Client(api_key="c7a22taad3iel691s2sg")

# %%
with open("all_stocks.json") as outfile:
    all_stocks = json.load(outfile)
all_symbols = []
for i in all_stocks:
    all_symbols.append(i.split('.')[0])
# print(len(all_symbols))
all_symbols = list(set(all_symbols))
# print(len(all_symbols))
# all_symbols
    # ticker =
# all_symbols


# %%
class Features():
    """

    """
    def __init__(self, symbol):
        self.ticker = yf.Ticker(symbol)
        self.symbol = symbol
        self.information = {}
        # changed ebit to net income
        try:
            self.ebit = self.ticker.financials.loc['Net Income']
            self.date_last_company_informations = self.ebit.index
        except:
            self.ebit = None
            self.date_last_company_informations = None


    def get_information(self):
        if self.ebit is None:
            return self.ticker
        else:
            self.information = self.general_informations()
            # data = ({"recommandation_trends":self.recommandation_trends()},{"ek_quote": self.ek_quote()}, {"earnings_growth":
            #         self.earnings_growth()}, {"revenue_growth": self.revenue_growth()}, {"kgv": self.kgv()}, {"kuv":
            #         self.kuv()}, {"cashFlow": self.cashFlow()})
            cash_flow = self.cashFlow()
            cash_flow.name = "cashFlow"
            ek_quote = self.ek_quote()
            ek_quote.name = "EK_Quote"
            cash_flow = pd.DataFrame(cash_flow)
            ek_quote = pd.DataFrame(ek_quote)
            # data = #(cash_flow, ek_quote,
            data = (cash_flow, ek_quote, self.recommandation_trends(), self.earnings_growth(), self.revenue_growth(),
                    self.kgv(), self.kuv(), self.beta())
            for i in data:
                # for key, value in i.items():
                    # while(len(value) < len(self.information.index)):
                    #     timestamp_temp = self.information.index[len(value)]
                    #     print(key)
                    #     value.loc[timestamp_temp] = [0]
                    # self.information.insert(0, key, value)
                # i = i.set_index(self.information.index[:len(i)])
                # i = i.set_index(self.date_last_company_informations[:len(i)])

                try:
                    self.information = self.information.join(i, how='outer')
                except ValueError:
                    print(f"Not able to connect information for {self.symbol}")
            print(self.information)
            return {self.symbol: self.information}

    def recommandation_trends(self):
        # Feature 1: Analystenempfehlungen
        recommandation_trends = self.ticker.recommendations  #finnhub_client.recommendation_trends(self.symbol)
        if recommandation_trends is not None:
            recommandation_trends = recommandation_trends.replace(to_replace = "Overweight", value = "Buy")
            recommandation_trends = recommandation_trends.replace(to_replace="Underweight", value="Sell")
            recommandation_trends = recommandation_trends.replace(to_replace="Market Outperform", value="Strong Buy")
            recommandation_trends = recommandation_trends.replace(to_replace="Equal-Weight", value="Neutral")
            recommandation_trends = recommandation_trends.replace(to_replace="equal-Weight", value="Neutral")
            recommandation_trends = recommandation_trends.replace(to_replace="Reduce", value="Sell")
            recommandation_trends = recommandation_trends.replace(to_replace="Sector Outperform", value="Buy")
            recommandation_trends = recommandation_trends.replace(to_replace="Outperform", value="Strong Buy")
            recommandation_trends = recommandation_trends.replace(to_replace = "Long-term Buy", value = "Buy")
            recommandation_trends = recommandation_trends.replace(to_replace = "Long-Term Buy", value = "Buy")
            recommandation_trends = recommandation_trends.replace(to_replace = "Market Perform", value = "Neutral")
            recommandation_trends = recommandation_trends.replace(to_replace = "Positive", value = "Buy")
            recommandation_trends = recommandation_trends.replace(to_replace = "Negative", value = "Sell")

            recommandations_aggregated = []
            for i in range(3):
                cumulated_score = 0
                starttime = self.date_last_company_informations[i+1]
                endtime = self.date_last_company_informations[i]
                times_reports = recommandation_trends.index
                recommandation_trends_temp = recommandation_trends[recommandation_trends.index >= starttime]
                recommandation_trends_temp = recommandation_trends_temp[recommandation_trends_temp.index <= endtime]
                if recommandation_trends_temp.empty:
                    score = 0
                else:
                    current_recommadtion = list(recommandation_trends_temp['To Grade'])
                    # score = current_recommadtion.value_counts()
                    # recommandation_trends_temp = [j for j in recommandation_trends['To Grade'] if recommandation_trends.index()
                    #                               >= starttime and recommandation_trends.index()  <= endtime]
                    # current_recommadtion = recommandation_trends_temp
                    score = current_recommadtion.count('Strong Buy')*2 + current_recommadtion.count('Buy') - \
                        current_recommadtion.count('Sell') - current_recommadtion.count('Strong Sell')*2
                    # print(recommandation_trends_temp['To Grade'])
                    score /= len(recommandation_trends_temp.index)
                        # current_recommadtion.count('Strong Buy') + current_recommadtion.count('Buy') +\
                        #      current_recommadtion.count('Sell') + current_recommadtion.count('Strong Sell')
                recommandations_aggregated.append(score)
        else:
            recommandations_aggregated = [None, None, None]
        recommandation = pd.DataFrame(recommandations_aggregated, columns=["recommandationScore"])
        recommandation = recommandation.set_index(self.date_last_company_informations[:3])
        return recommandation

    # for i in range(len(recommandation_trends) // 4):
    #     cumulated_score = 0
    #     for j in range(3):
    #         current_recommadtion = recommandation_trends[i * 3 + j]
    #         score = current_recommadtion['Strong Buy'] * 2 + current_recommadtion['Buy'] - \
    #                 current_recommadtion['Sell'] - current_recommadtion['Strong Sell'] * 2
    #         score /= current_recommadtion['Strong Buy'] + current_recommadtion['Buy'] - \
    #                  current_recommadtion['Sell'] + current_recommadtion['Strong Sell']
    #         cumulated_score += score
    #     recommandations_aggregated.update(
    #         {recommandation_trends[i * 3]['period']: cumulated_score / 3})
    # return recommandations_aggregated
    #
    def ek_quote(self):
        """

        :return:
        """
    # Feature 2: EK Quote
        return (self.ticker.balance_sheet.loc['Total Assets'] -
                self.ticker.balance_sheet.loc['Total Liab'])/self.ticker.balance_sheet.loc['Total Assets']


    def earnings_growth(self):
        """

        :return:
        """
        # Feature 3: Gewinnwachstum
        ebit_growth = pd.DataFrame(columns=["ebit_growth"])
        for i in range(1, len(self.ebit)):
            time = self.date_last_company_informations[i-1]
            try:
                ebit_growth.loc[time] = self.ebit.iloc[i-1]/self.ebit.iloc[i]
            except:
                ebit_growth
        return ebit_growth

    def revenue_growth(self):
        # Feature 4: Umsatzwachstum
        revenue = self.ticker.financials.loc['Total Revenue']
        revenue_growth = pd.DataFrame(columns=["revenue_growth"])
        for i in range(1, len(self.ebit)):
            time = self.date_last_company_informations[i-1]
            revenue_growth.loc[time] = revenue.iloc[i-1]/revenue.iloc[i]
        return revenue_growth

    def kgv(self):
        # Feature 5: Kurs-Gewinn-Verhältnis
        kgv = pd.DataFrame(columns=["kgv"])
        for i in range(len(self.ebit)):
            time = self.date_last_company_informations[i]
            share_price = self.ticker.history(start=time, end=time)
            share_price = list(share_price.loc[:, 'Close'])
            counter = 0
            time_new = time
            while(len(share_price) != 1 and counter != 10):
                time_new += timedelta(days=1)
                share_price = self.ticker.history(start=time_new, end=time_new)
                share_price = list(share_price.loc[:, 'Close'])
                counter += 1
            if len(share_price) == 1:
                market_cap = share_price[0]*self.ticker.info['sharesOutstanding']
            else:
                market_cap = -1
            # kgv.update({time: market_cap/self.ebit.iloc[i]})
            kgv.loc[time] = market_cap/self.ebit.iloc[i]
        return kgv


    def kuv(self):
        # Feature 6: Kurs-Umsatz-Verhältnis
        revenue = self.ticker.financials.loc['Total Revenue']
        kuv = pd.DataFrame(columns=["kuv"])
        for i in range(len(revenue)):
            time = revenue.index[i]
            share_price = self.ticker.history(start=time, end=time)
            share_price = list(share_price.loc[:, 'Close'])
            counter = 0
            time_new = time
            while (len(share_price) != 1 and counter != 10):
                time_new += timedelta(days=1)
                share_price = self.ticker.history(start=time_new, end=time_new)
                share_price = list(share_price.loc[:, 'Close'])
                counter += 1
            if len(share_price) == 1:
                market_cap = share_price[0] * self.ticker.info['sharesOutstanding']
            else:
                market_cap = -1
            # kgv.update({time: market_cap/self.ebit.iloc[i]})
            kuv.loc[time] = market_cap/revenue.iloc[i]
        return kuv

    def general_informations(self):
        # Feature 7: Land (+ Sektor für Vorklassifizierung)
        # company_profile = finnhub_client.company_profile2(symbol=self.symbol)
        company_profile = self.ticker.info
        country = company_profile['country']
        sector = company_profile['sector']
        name = company_profile['longName']
        longBusinessSummary = company_profile['longBusinessSummary']
        data = [[j for j in (name, sector, country, longBusinessSummary)] for i in range(4)]
        general_information = pd.DataFrame(data, columns=["name", "sector", "country", "longBusinessSummary"])
        general_information = general_information.set_index(self.date_last_company_informations)
        return general_information #{"country": country, "sector": sector, "name": name, "description": longBusinessSummary}

    def cashFlow(self):
        # Feature 8: Cash Flow: change in cash
        return self.ticker.cashflow.loc['Change In Cash']
    #
    # def beta1(self):
    #     betas = []
    #     for i in range(len(self.date_last_company_informations)-1):
    #         start = self.date_last_company_informations[i]
    #         if i == 0:
    #             end = date.today()
    #         else:
    #             end = self.date_last_company_informations[i-1]
    #         data = yf.download(f"^GSPC {self.symbol}", start=start, end=end, interval='1mo')
    #         data = data.iloc[1:, :2]
    #         # data = data.pct_change()
    #         data = data.dropna()
    #         df = data
    #         X = df.values[:, [0]]
    #         b = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(df.values[:, 1:])
    #         betas.append(b)
    #     return pd.Series(b[1], df.columns[1:], name='Beta')

    def beta(self):
        betas = pd.DataFrame(columns=["beta_following_period"])
        for i in range(len(self.date_last_company_informations) - 1):
            start = self.date_last_company_informations[i]
            if i == 0:
                end = date.today()
            else:
                end = self.date_last_company_informations[i - 1]
            data = yf.download(f"^GSPC {self.symbol}", start=start, end=end)#, interval='1d')
            data = data.iloc[:, :2]
            data = data.pct_change()
            data = data.dropna()
            cov = data.cov().iloc[0,1]
            beta = cov/data.iloc[:,1].var()
            betas.loc[start] = beta
        return betas
        #
        # # first column is the Sales
        # X = df.values[:, [-1]]
        # # prepend a column of ones for the intercept
        # X = np.concatenate([np.ones_like(X), X], axis=1)
        # # matrix algebra
        # b = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(df.values[:, :-1])
        # return pd.Series(b[1], df.columns[:-1], name='Beta')

if __name__ == '__main__':
    start_time = time.time()
    data = {}
    not_enough_information = []
    not_included_in_yfinance = []
    start = 0
    end = 0
    all_symbols = all_symbols[::500]
    # all_symbols = ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'GOOGL', 'GOOG', 'FB', 'NVDA', 'UNH', 'JPM', 'JNJ', 'HD', 'BRK.B', 'PG', 'NESN', 'V', 'BAC', 'ASML', 'PFE', 'MA']
    for i in all_symbols:
        print(f"start calculating symbol: {i}")
        test = Features(i)
        try:
            value = test.get_information()
            if type(value)==dict:
                symbol = list(value.keys())[0]
                value = list(value.values())[0]
                value = value.to_json()
                data.update({symbol: value})
        except:
            not_enough_information.append(i)
        else:
            not_included_in_yfinance.append(i)

    with open(f"data/{end}_test_data_classification.json", "w") as outfile:
        json.dump(data, outfile)
    print(len(not_enough_information+not_included_in_yfinance))
    with open(f"data/{end}_test_data_classification_not_included.json", "w") as outfile:
        not_enough_information = {"not included": not_included_in_yfinance, "not enough information":
            not_enough_information}
        json.dump(not_enough_information, outfile)
    time_taken = time.time() - start_time
    print(f"Needed {time_taken} seconds to calculate {len(data)} valid datapoints and "
          f"{len(not_enough_information)} invalid")