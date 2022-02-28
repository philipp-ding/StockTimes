# import libaries
import json
from tkinter import *
from tkinter import messagebox
import os
import sys
import pandas as pd
from PIL import Image, ImageTk
from threading import Thread

# internal librarys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../classification'))
from sentiment_analysis_stocks_v1 import run_all_functions
from historical_beta_approach import aktualisieren

# import predicted dataframe
data = pd.read_csv(r'Data\predicted_daframe.csv')
data = data.sample(frac=1)  # shuffle data to get random stocks

# get data of sentiment analysis
with open("Data\json_for_frontend.json") as json_for_frontend:
    datasentiment = json.load(json_for_frontend)  # load json file data
    sentiment = list(datasentiment.keys())[0]  # get market sentiment
    sectors = list(datasentiment.values())[0]  # get market sectors
print(sentiment)  # check market sentiment
print(sectors)  # check market sector of current sentiment

root = Tk()  # get tkinter window
root.title("StockTimes")  # sets title of the app
root.iconbitmap("Images\stocktimes.ico")  # import the StockTimes icon

# Welcome text of the StockTimes App
framewelcome = LabelFrame(root, padx=327,
                          pady=10)  # create a label for the welcome text
framewelcome.grid(row=0, column=0,
                  columnspan=3)  # set label on a specific space of the grid
myLabel = Label(
    framewelcome,
    text="Herzlich willkommen bei StockTimes!",
    font="Verana 15 bold")  # set welcome text that is displayed & the size
myLabel.grid(row=0, column=0, columnspan=3)  # set position of welcome text

# risk appetite
framerisiko = LabelFrame(root,
                         text="Wie viel Risiko wollen Sie eingehen?",
                         padx=150,
                         pady=75,
                         font="Verana 10 bold")  # label to ask for risik

framerisiko.grid(row=1, column=0, columnspan=1)  # sets position of the label

# StockTimes Logo
# Quit Button
framequit = LabelFrame(root, padx=75, pady=5)  # create frame for button
framequit.grid(sticky="E", row=2, column=2, columnspan=1)  # set position
# button to close & end app
button_quit = Button(framequit, text="StockTimes verlassen",
                     command=root.quit)  # button to close frame
button_quit.grid(sticky="E", row=2, column=2, columnspan=1)  # sets position

def update_all_data_with_threading():
    """starts to reload sentiment analysis and download + predicts beta
    use threading to run functions in the background
    :return:
    """
    thread1 = Thread(target=reload)
    thread1.daemon = True
    thread1.start()
    thread2 = Thread(target=aktualisieren)
    thread2.daemon = True
    thread2.start()

# reload smiley if market mood changes // refresh sentiment analysis
def reload():
    run_all_functions(
    )  # runs the sentimentanalysis to find changes in the market
    global marktstimmung
    global stimmung
    with open("Data\json_for_frontend.json") as json_for_frontend:
        datasentiment = json.load(
            json_for_frontend)  # load data of the sentimentanalysis
        sentiment = list(datasentiment.keys())[
            0]  # get marketmood // positiv, neutral, negativ
        sectors = list(datasentiment.values())[
            0]  # get sectors matching to current market situation

        print(sentiment)
        print(sectors)

    # market mood of sentiment analysis
    stimmung = str(sentiment)  # format the market mood to a string
    # smiley for current market sentiment
    if stimmung == 'positive':
        marktstimmungbild = ImageTk.PhotoImage(
            Image.open("Images\Smileyhappy.png"))  # smiles if positive
    elif stimmung == 'neutral':
        marktstimmungbild = ImageTk.PhotoImage(
            Image.open(
                "Images\Smileyneutral.png"))  # neutral smiley if neutral
    elif stimmung == 'negative':
        marktstimmungbild = ImageTk.PhotoImage(
            Image.open(
                "Images\Smileysad.png"))  # sad smiley if negative market mood

    marktstimmungbildlabel.configure(
        image=marktstimmungbild)  # change smiley if market mood switches
    marktstimmungbildlabel.photo = marktstimmungbild  # smiley gets out of cache
    marktstimmungbildlabel.grid(row=0, column=3,
                                columnspan=1)  # set position of market mood
    global marktstimmungsbild


# Reload Button
framequit = LabelFrame(root, padx=61, pady=5)  # create frame for button
framequit.grid(sticky="W", row=2, column=2, columnspan=1)  # sets position
button_reload = Button(framequit,
                       text="Marktstimmung ermitteln",
                       command=update_all_data_with_threading)  # button to refresh current market mood
button_reload.grid(sticky="W", row=2, column=2,
                   columnspan=1)  # set button into specific place


def print_risiko(v):
    # Labeframe for current risik user takes
    # v is value of the slider
    if v == '0':
        risiko_text = 'Absicherung'  #  no risk
    elif v == '1':
        risiko_text = 'Sehr geringes Risiko'
    elif v == '2':
        risiko_text = 'Geringes Risiko'
    elif v == '3':
        risiko_text = 'Mittleres Risiko'
    elif v == '4':
        risiko_text = 'Hohes Risiko'
    elif v == '5':
        risiko_text = 'Sehr hohes Risiko'
    risiko_neigung = 'Aktuelle Risikoneigung: {}'.format(
        risiko_text)  #  set labeframe text with current slider value (v)
    l.config(text=risiko_neigung,
             font="Verana 9 bold")  # set font size of labeframe text


# slider to select risk
slider = Scale(framerisiko,
               from_=0,
               to=5,
               orient=HORIZONTAL,
               command=print_risiko)  # slider for risk setting
slider.grid(row=1, column=0, columnspan=1)  # set place of slider in grid

l = Label(root)
l.grid(row=2, column=0)  # sets position

# market mood label
marktstimmung = Label(framerisiko, text="Marktstimmung:",
                      font="Verana 9 bold")  # set text & font of header

marktstimmung.grid(row=0, column=0, columnspan=1)  # sets position

stimmung = str(sentiment)  # market mood detected by sentiment analysis
# change smiley if market mood changes
if stimmung == 'positive':
    marktstimmungbild = ImageTk.PhotoImage(
        Image.open("Images\Smileyhappy.png"))  # smiles if positive
elif stimmung == 'neutral':
    marktstimmungbild = ImageTk.PhotoImage(
        Image.open("Images\Smileyneutral.png"))  # neutral smile if neutral
elif stimmung == 'negative':
    marktstimmungbild = ImageTk.PhotoImage(
        Image.open("Images\Smileysad.png"))  # sad if negative market mood

marktstimmungbildlabel = Label(
    framerisiko, image=marktstimmungbild)  # adjust smiley to new market mood
marktstimmungbildlabel.grid(row=0, column=3,
                            columnspan=1)  # set position of smiley

# our suggestions // right half of Tkinter Frame
framevorschlag = LabelFrame(
    root, text="Unsere Vorschläge", padx=150, pady=27,
    font="Verana 10 bold")  # LabeFrame for our suggestions
framevorschlag.configure(height=281,
                         width=550)  # set height of LabeFrame content
framevorschlag.grid(row=1, column=2, columnspan=1)  # set positions
framevorschlag.grid_propagate(0)  # set position
# five empty labels to insert stock predictions later
vorschlag1 = Label(framevorschlag, text="", pady=6.5, anchor='w')
vorschlag1.grid(row=2, column=2, columnspan=1)
vorschlag2 = Label(framevorschlag, text="", pady=6.5, anchor='w')
vorschlag2.grid(row=3, column=2, columnspan=1)
vorschlag3 = Label(framevorschlag, text="", pady=6.5, anchor='w')
vorschlag3.grid(row=4, column=2, columnspan=1)
vorschlag4 = Label(framevorschlag, text="", pady=6.5, anchor='w')
vorschlag4.grid(row=5, column=2, columnspan=1)
vorschlag5 = Label(framevorschlag, text="", pady=6.5, anchor='w')
vorschlag5.grid(row=6, column=2, columnspan=1)
vorschlag6 = Label(framevorschlag, text="", pady=6.5, anchor='w')
vorschlag6.grid(row=7, column=2, columnspan=1)


def vorschlaege():
    # function to fill five empty labels with stock predictions
    risiko = slider.get()  # to get risk selected by the user
    global framevorschlag
    global vorschlag1
    global vorschlag2
    global vorschlag3
    global vorschlag4
    global vorschlag5
    global sectors
    try:
        # get different Stocks -> not everytime the same
        # shuffeld dataframe
        data = pd.read_csv(
            'Data\predicted_daframe.csv'
        )  # load predicted data // stocks customized to risik
        data = data.sample(frac=1)  # shuffels data
        # required data out of dataframe
        df = data.iloc[:, 1:]
        df.loc[df['sector'].isin(
            sectors)]  # gets data with predicted market sector
        # get data for each risk appetite
        risiko0 = df[df['prediction'] == 0]  # stock data with no risk
        risiko0 = risiko0['name'][:5].to_list()  # first 5 stock predictions
        #
        risiko1 = df[df['prediction'] ==
                     1]  # stock data predictions with little risk
        risiko1 = risiko1['name'][:5].to_list()  # first 5 stock predictions
        #
        risiko2 = df[df['prediction'] ==
                     2]  # stock data predictions with more risk
        risiko2 = risiko2['name'][:5].to_list()  # first 5 stock predictions
        #
        risiko3 = df[df['prediction'] ==
                     3]  # stock data prediction middle risk
        risiko3 = risiko3['name'][:5].to_list()  # first 5 stock predictions
        #
        risiko4 = df[df['prediction'] == 4]  # stock data prediction high risk
        risiko4 = risiko4['name'][:5].to_list()  # first 5 stock predictions
        #
        risiko5 = df[df['prediction'] == 5]  # stock data prediction highest risk
        risiko5 = risiko5['name'][:5].to_list()  # first 5 stock predictions
        #
        # if enough five stock predictions // check if every risk has enough stocks to fill the labels
        if len(risiko1) < 5:
            # if data is not enough to fill 5 stock suggestions
            data = pd.read_csv('Data\predicted_daframe.csv'
                               )  # dataframe with predicted stocks
            data = data.sample(frac=1)  # shuffle dataframe
            name_1 = df['name'].to_list()  # stock names into a list
            # data with little risk
            risiko_1 = data[data['prediction'] == 1]
            risiko_1 = risiko_1[
                ~risiko_1['name'].isin(name_1)]  # removes duplicates of list
            risiko_1 = risiko_1['name'].to_list()  # list with stock names
            risiko1.extend(risiko_1)  # add data to prior data
        elif len(risiko2) < 5:
            # if data is not enough to fill 5 Labeframes
            data = pd.read_csv('Data\predicted_daframe.csv')
            data = data.sample(frac=1)  # shuffle data
            name_2 = df['name'].to_list()
            risiko_2 = data[data['prediction'] ==
                            2]  # look for stocks with more risk appetite
            risiko_2 = risiko_2[
                ~risiko_2['name'].isin(name_2)]  # removes duplicates of list
            risiko_2 = risiko_2['name'].to_list()
            risiko2.extend(risiko_2)  # add data to existing data frame
        elif len(risiko3) < 5:
            # if data is not enough to fill 5 Labeframes
            data = pd.read_csv('Data\predicted_daframe.csv')
            data = data.sample(frac=1)
            name_3 = df['name'].to_list()
            risiko_3 = data[data['prediction'] == 3]
            risiko_3 = risiko_3[
                ~risiko_3['name'].isin(name_3)]  # removes duplicates of list
            risiko_3 = risiko_3['name'].to_list()
            risiko3.extend(risiko_3)  # add data to prior data
        elif len(risiko4) < 5:
            # if data is not enough to fill 5 Labeframes
            data = pd.read_csv('Data\predicted_daframe.csv')
            data = data.sample(frac=1)
            name_4 = df['name'].to_list()
            risiko_4 = data[data['prediction'] == 4]
            risiko_4 = risiko_4[
                ~risiko_4['name'].isin(name_4)]  # removes duplicates of list
            risiko_4 = risiko_4['name'].to_list()
            risiko4.extend(risiko_4)  # add data to prior data
        elif len(risiko5) < 5:
            # if data is not enough to fill 5 Labeframes
            data = pd.read_csv(
                'Data\predicted_daframe.csv'
            )  # dataframe with predicted stocks and risk appetite
            data = data.sample(frac=1)  # shuffle data
            name_5 = df['name'].to_list()  # list of predicted stock names
            risiko_5 = data[data['prediction'] == 5]
            risiko_5 = risiko_5[
                ~risiko_5['name'].isin(name_5)]  # removes duplicates of list
            risiko_5 = risiko_5['name'].to_list()
            risiko5.extend(risiko_5)  # add data to prior data
        elif len(risiko0) < 5:
            # if data is not enough to fill 5 Labeframes
            data = pd.read_csv('Data\predicted_daframe.csv')
            data = data.sample(frac=1)  # shuffle data
            name_0 = df['name'].to_list()  # list of stock names
            risiko_0 = data[data['prediction'] ==
                            0]  # stocks with no risk appetite
            risiko_0 = risiko_0[
                ~risiko_0['name'].isin(name_0)]  # removes duplicates of list
            risiko_0 = risiko_0['name'].to_list()  # list of stock names
            risiko0.extend(risiko_0)  # add stock suggestions to existing data
        # get five stock names out of the data to fill empty labeframes for each risk class
        if risiko == 1:
            risiko_klasse = 'eine sehr geringe Risikoneigung'  # set shown risk appetite of stock suggestions
            # five stock suggestions
            aktie_1 = risiko1[0]
            aktie_2 = risiko1[1]
            aktie_3 = risiko1[2]
            aktie_4 = risiko1[3]
            aktie_5 = risiko1[4]
        elif risiko == 2:
            risiko_klasse = 'eine geringe Risikoneigung'  # set shown risk appetite of stock suggestions
            # five stock suggestions
            aktie_1 = risiko2[0]
            aktie_2 = risiko2[1]
            aktie_3 = risiko2[2]
            aktie_4 = risiko2[3]
            aktie_5 = risiko2[4]
        elif risiko == 3:
            risiko_klasse = 'eine mittlere Risikoneigung'  # set shown risk appetite of stock suggestions
            # five stock suggestions
            aktie_1 = risiko3[0]
            aktie_2 = risiko3[1]
            aktie_3 = risiko3[2]
            aktie_4 = risiko3[3]
            aktie_5 = risiko3[4]
        elif risiko == 4:
            risiko_klasse = 'eine hohe Risikoneigung'  # set shown risk appetite of stock suggestions
            # five stock suggestions
            aktie_1 = risiko4[0]
            aktie_2 = risiko4[1]
            aktie_3 = risiko4[2]
            aktie_4 = risiko4[3]
            aktie_5 = risiko4[4]
        elif risiko == 5:
            risiko_klasse = 'eine sehr hohe Risikoneigung'  # set shown risk appetite of stock suggestions
            # five stock suggestions
            aktie_1 = risiko5[0]
            aktie_2 = risiko5[1]
            aktie_3 = risiko5[2]
            aktie_4 = risiko5[3]
            aktie_5 = risiko5[4]
        elif risiko == 0:
            risiko_klasse = 'keine Risikoneigung'  # set shown risk appetite of stock suggestions
            # five stock suggestions
            aktie_1 = risiko0[0]
            aktie_2 = risiko0[1]
            aktie_3 = risiko0[2]
            aktie_4 = risiko0[3]
            aktie_5 = risiko0[4]
        else:
            print('')
        framevorschlag.config(text="Unsere Vorschäge für {}".format(
            risiko_klasse))  # show selected risk appetite of user
        # list of suggested stocks
        vorschlag1.config(text="{}".format(aktie_1))
        vorschlag2.config(text="{}".format(aktie_2))
        vorschlag3.config(text="{}".format(aktie_3))
        vorschlag4.config(text="{}".format(aktie_4))
        vorschlag5.config(text="{}".format(aktie_5))

    except:
        try:
            data = pd.read_csv('Data\predicted_daframe.csv'
                               )  # dataframe with predicted stocks
            data = data.sample(frac=1)  # shuffle data
            # important parts of predicted data frame
            df = data.iloc[:, 1:]
            df.loc[df['sector'].isin(
                sectors
            )]  # get stocks fitting to sectors detected in sentiment analysis
            #
            if risiko == 0:
                risiko_klasse = 'keine Risikoneigung'
                risiko0 = df[df['prediction'] == 0]
                risiko0 = risiko0['name'][:5].to_list()
                if len(risiko0) < 5:
                    data = pd.read_csv('Data\predicted_daframe.csv')
                    data = data.sample(frac=1)  # shuffle data
                    name_0 = df['name'].to_list()  # list of stock names
                    risiko_0 = data[data['prediction'] == 0]
                    risiko_0 = risiko_0[~risiko_0['name'].isin(
                        name_0)]  # removes duplicates of list
                    risiko_0 = risiko_0['name'].to_list(
                    )  # list of stock names
                    aktien = risiko0.extend(risiko_0)  # add data to prior data
                aktien = risiko0
            elif risiko == 1:
                risiko_klasse = 'eine sehr geringe Risikoneigung'
                risiko1 = df[df['prediction'] == 1]  # stocks with little risk
                risiko1 = risiko1['name'][:5].to_list(
                )  # get first five stock names
                if len(risiko1) < 5:
                    data = pd.read_csv('Data\predicted_daframe.csv'
                                       )  # dataframe with predicted stocks
                    data = data.sample(frac=1)  # shuffle data
                    name_1 = df['name'].to_list()  # list of stock names
                    risiko_1 = data[data['prediction'] == 1]
                    risiko_1 = risiko_1[~risiko_1['name'].isin(
                        name_1)]  # removes duplicates of list
                    risiko_1 = risiko_1['name'].to_list(
                    )  # list of stock names
                    aktien = risiko1.extend(risiko_1)  # add data to prior data
                aktien = risiko1  # stock names to fill labels
            elif risiko == 2:
                risiko_klasse = 'eine geringe Risikoneigung'
                risiko2 = df[df['prediction'] == 2]  # stocks with little risk
                risiko2 = risiko2['name'][:5].to_list(
                )  # first five stock names
                if len(risiko2) < 5:
                    data = pd.read_csv('Data\predicted_daframe.csv')
                    data = data.sample(frac=1)  # shuffle data
                    name_2 = df['name'].to_list()  # names of stocks
                    risiko_2 = data[data['prediction'] == 2]
                    risiko_2 = risiko_2[~risiko_2['name'].isin(
                        name_2)]  # removes duplicates of list
                    risiko_2 = risiko_2['name'].to_list()
                    aktien = risiko2.extend(risiko_2)  # add data to prior data
                aktien = risiko2  # stock names to fill labels
            elif risiko == 3:
                risiko_klasse = 'eine mittlere Risikoneigung'
                risiko3 = df[df['prediction'] == 3]  # stocks with middle risk
                risiko3 = risiko3['name'][:5].to_list()  # first five stocks
                if len(risiko3) < 5:
                    data = pd.read_csv('Data\predicted_daframe.csv')
                    data = data.sample(frac=1)  # shuffle data
                    name_3 = df['name'].to_list()
                    risiko_3 = data[data['prediction'] == 3]
                    risiko_3 = risiko_3[~risiko_3['name'].isin(
                        name_3)]  # removes duplicates of list
                    risiko_3 = risiko_3['name'].to_list()
                    aktien = risiko3.extend(risiko_3)  # add data to prior data
                aktien = risiko3  # stock names to fill labels
            elif risiko == 4:
                risiko_klasse = 'eine hohe Risikoneigung'
                risiko4 = df[df['prediction'] == 4]  # stocks with high risk
                risiko4 = risiko4['name'][:5].to_list()
                if len(risiko4) < 5:
                    data = pd.read_csv('Data\predicted_daframe.csv')
                    data = data.sample(frac=1)  # shuffle data
                    name_4 = df['name'].to_list()
                    risiko_4 = data[data['prediction'] == 4]
                    risiko_4 = risiko_4[~risiko_4['name'].isin(
                        name_4)]  # removes duplicates of list
                    risiko_4 = risiko_4['name'].to_list()
                    aktien = risiko4.extend(risiko_4)  # add data to prior data
                aktien = risiko4  # stock names to fill labels
            elif risiko == 5:
                risiko_klasse = 'eine sehr hohe Risikoneigung'
                risiko5 = df[df['prediction'] ==
                             5]  # stocks with higher risk appetite
                risiko5 = risiko5['name'][:5].to_list()
                if len(risiko5) < 5:
                    data = pd.read_csv('Data\predicted_daframe.csv')
                    data = data.sample(frac=1)  # shuffle stock names
                    name_5 = df['name'].to_list()  # list of stock names
                    risiko_5 = data[data['prediction'] ==
                                    5]  # stocks with higher risk appetite
                    risiko_5 = risiko_5[~risiko_5['name'].isin(
                        name_5)]  # removes duplicates of list
                    risiko_5 = risiko_5['name'].to_list(
                    )  # list of stock names
                    aktien = risiko5.extend(risiko_5)  # add data to prior data
                aktien = risiko5  # stock names to fill labels
            if len(aktien) < 5:
                # not enough stock predictions to fill five suggestions
                n = len(aktien)  # number of stock predictions
                aktien = aktien[:n + 1]  # get all available stock predictions

                if n == 0:
                    framevorschlag.config(
                        text="Unsere Vorschäge für {}".format(risiko_klasse)
                    )  # show selected risk appetite of user
                    # labels empty because no stocks for risk class & class sector exists
                    vorschlag1.config(text="")
                    vorschlag2.config(text="")
                    vorschlag3.config(text="")
                    vorschlag4.config(text="")
                    vorschlag5.config(text="")

                    messagebox.showerror(
                        'Achtung',
                        'Es sind keine Aktien vorhanden ! Wählen Sie eine andere Risikoneigung aus'
                    )  # error message if no stocks are available

                    #

                elif n == 1:
                    framevorschlag.config(
                        text="Unsere Vorschäge für {}".format(risiko_klasse)
                    )  # show selected risk appetite of user
                    # list of suggested stocks
                    vorschlag1.config(text="{}".format(aktien[0]))
                    vorschlag2.config(text="")
                    vorschlag3.config(text="")
                    vorschlag4.config(text="")
                    vorschlag5.config(text="")

                elif n == 2:
                    framevorschlag.config(
                        text="Unsere Vorschäge für {}".format(risiko_klasse)
                    )  # show selected risk appetite of user
                    # list of suggested stocks
                    vorschlag1.config(text="{}".format(aktien[0]))
                    vorschlag2.config(text="{}".format(aktien[1]))
                    vorschlag3.config(text="")
                    vorschlag4.config(text="")
                    vorschlag5.config(text="")

                elif n == 3:
                    framevorschlag.config(text="Unsere Vorschäge {}".format(
                        risiko_klasse))  # show selected risk appetite of user
                    # list of suggested stocks
                    vorschlag1.config(text="{}".format(aktien[0]))
                    vorschlag2.config(text="{}".format(aktien[1]))
                    vorschlag3.config(text="{}".format(aktien[2]))
                    vorschlag4.config(text="")
                    vorschlag5.config(text="")

                elif n == 4:
                    framevorschlag.config(
                        text="Unsere Vorschäge für {}".format(risiko_klasse)
                    )  # show selected risk appetite of user
                    # list of suggested stocks
                    vorschlag1.config(text="{}".format(aktien[0]))
                    vorschlag2.config(text="{}".format(aktien[1]))
                    vorschlag3.config(text="{}".format(aktien[2]))
                    vorschlag4.config(text="{}".format(aktien[3]))
                    vorschlag5.config(text="")

        except:
            # if no data is available for risk appetite and sectors (of market mood)
            framevorschlag.config(text="Unsere Vorschäge für {}".format(
                risiko_klasse))  # show selected risk appetite of user
            # fill suggestions with no stock predictions
            vorschlag1.config(text="")
            vorschlag2.config(text="")
            vorschlag3.config(text="")
            vorschlag4.config(text="")
            vorschlag5.config(text="")
            messagebox.showerror(
                'Achtung',
                'Es sind keine Aktien vorhanden ! Wählen Sie eine andere Risikoneigung aus'
            )  # error message if no data exists


# button to confirm selected risk class
risikobutton = Button(framerisiko,
                      text="Eingabe Bestätigen",
                      command=vorschlaege)  # button to confirm risk appetite
risikobutton.grid(row=2, column=0,
                  columnspan=1)  # set position of confirmation button

root.resizable(False, False)  # window can not change its size

root.mainloop()  # loop of tkinter application
