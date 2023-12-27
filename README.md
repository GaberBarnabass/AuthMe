# AuthMe
per info leggere doc.pdf

a causa della grande mole di dati ottenuti dall'estrazione delle feature per i diversi tempi di autenticazione si è deciso di non includere tutte le feature estratte.
Tutta via queste possono essere calcolate nuovamente:
1) scaricare i due dataset  
   BrainRun - https://zenodo.org/records/2598135  
   HMOG - https://hmog-dataset.github.io/hmog/  
2) una volta scaricati i dataset   
  a) tutte le cartelle di HMOG vanno spostate in HOMG -> raw sensors -> raw data  
  b) per BrainRun invece, prendere i dati contenuti nella cartella sensor_data e dividere i file   
     json in cartelle: una per ogni utente e popolare la cartella BrainRun -> raw sensor -> raw data  

   
```
+---data
    +---BrainRun
    ª   +---2 seconds
    ª   +---3 seconds
    ª   +---4 seconds
    ª   +---5 seconds
    ª   +---raw sensors
    ª       +---accelerometer       
    ª       +---gyroscope       
    ª       +---raw data
    ª           
    +---HMOG
    ª   +---0.2 seconds
    ª   +---0.5 seconds
    ª   +---1 seconds
    ª   +---2 seconds
    ª   +---raw sensors
    ª       +---accelerometer
    ª       +---gyroscope
    ª       +---raw data
```
