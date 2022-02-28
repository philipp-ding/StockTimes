
## Projektbeschreibung

Im Rahmen der Vorlesungseinheit "Fallstudie" wurde dieses Projekt entwickelt. Es beschäftigt sich mit einer Möglichkeit Anlegern zu helfen, eine Investitionsentscheidung zu treffen.

Unsere Aufgabe war es, einen Weg zu finden, Kunden bestmöglich zu beraten. Dabei sollte zum einen die persönliche Risikofreudigkeit
und zum anderen die aktuelle weltweite Entwicklung und die damit verbundene Marktstimmung berücksichtigt werden. Zusätzlich sollten diese beiden Faktoren dann
noch von einem Machine Learning-Modell unterstützt werden um eine optimale Entscheidung zu ermöglichen. Die Marktstimmung soll auf Basis verschiedener
Meinungen ermittelt werden. Für das Machine Learning-Modell sollen historische Daten sowie verschiedene Kennzahlen genutzt werden.
Insgesamt war es also das Ziel dieser Fallstudie, Kunden Aktien auf Basis ihrer individuell einstellbaren Risikoneigung und der aktuellen Marktstimmung
vorzuschlagen. Wie wir dabei vorgegangen sind wird im Laufe dieser Dokumentation erklärt.
	



## Installation des Programms:
```
- Lazy Version:
    ○ Pip install -r requirements.txt 
    ○ Frontend.py Datei des Ordners FrontendFallstudie in einer Python-Distribution öffnen und ausführen

- Empfohlene Version: 
  (Wenn eine TensorFlow-Version vorhanden ist, die nicht der benötigten Version 2.7 enstpricht, dann unbedingt, 
   ansonsten empfohlen wir ein virtual Environment (venv) zu erstellen)
        (1) Klonen des Github-Repository (Branch: main)
	(2) Öffnen der CMD und navigieren in den Git-Hub-Ordner
	(3) CMD-Command "python3.8 -m venv StockTimes" asuführen (erstellt die virtuell Environment, ersetzen von 3.8 durch die vorhandene Python-Version)
	(4) Navigieren in den Ordner und "Scripts\activate.bat" in CMD ausführen (aktiviert die virtuell Environment)
	(4) CMD-Command "Pip install -r requirements.txt" ausführen 
	(5) Frontend.py Datei des Ordners FrontendFallstudie in einer Python-Distribution
	(6) Auswählen des Python Interpreters aus dem Ordner Scripts
	(7) Nur bei Visual Studio: Navigieren in den Ordner FrontendFallstudie im Terminal, damit die Abhängigkeiten der Pfade passen
	(8) Programm ausführen

- Weitere Informationen zur virtual Environment: https://snarky.ca/a-quick-and-dirty-guide-on-how-to-install-packages-for-python/
```

## Beachten bei der Verwendung:
```
- Der Button "Marktstimmung ermitteln" kann verwendet werden um die aktuellen Aktienkurse herunterzuladen 
  und darauf aktuelle Vorhersagen auszuführen und um eine aktuelle Sentiment Analyse durchzuführen
- Beides findet im Hintergrund statt und beeinträchtigt die Nutzung der App nicht
- Bis aber die Ergebnisse angezeigt werden, werden folgende Zeitspannen benötigt:
	○ Sentiment Analyse: ca. 2 Minuten
	○ Risikoklassifizierung:  mind. 30 Minuten
- Wenn die App beendet wird, werden beide Prozesse abgebrochen --> App funktioniert weiterhin auf den vorherigen Daten![image](https://user-images.githubusercontent.com/78366154/155842675-35ce53c3-5af8-453c-98ec-695fbc113991.png)
- Das Programm hisorical_beta_approach aus dem Ordner classification kann ebenfalls ausgeführt werden per Doppelklick oder über die cmd. 
  Hierbei kann exemplarisch für ein paar Aktien ein neues Modell trainiert werden, oder auf einem bereits im trained_model Ordner vorhandenem Vorhersagen durchgeführt werden.
  Wenn ein neues Modell trainiert wurde, wird dieses und weitere Diagramme + Informationen über das Training im trained_model Ordner gespeichert. 
 ```
