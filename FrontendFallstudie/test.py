import sys, os

# internal librarys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../classification'))
from historical_beta_approach import aktualisieren

aktualisieren()