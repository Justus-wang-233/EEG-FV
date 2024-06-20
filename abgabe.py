# Nichts Ändern!!!
import client
# Ihr könnt nur nach 10 sekunden wieder eine submission zum Server machen.
# Dies ist als Schutz für euch gedacht,
# um das versehentliche doppelt Ausführen von Zellen zu verhindern.
import time
last_server_interaction = 0
cooldown = 10  # seconds

# Beispiel für uns (ersetzt den rechten Teil der ':' mit euren Angaben):
abgabe = {
    "team_name": "EEG-FV",
    "git_SSH": "git@github.com:Justus-wang-233/EEG-FV.git",
    "model": "model.json",
    "python_version": 3.8
         }

if time.time()-last_server_interaction < cooldown:
    print("please wait")
else:
    client.test_submission(abgabe)
    last_server_interaction = time.time()