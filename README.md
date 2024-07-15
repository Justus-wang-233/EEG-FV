Wir haben insgesamt zwei Modelle verwendet: 
**wbcknn_model.json** und **gbdt_model.json**
Auf GitHub gibt es immer noch model.json, aber deren Inhalt ist leer. 
Um diese Codezeile nicht zu ändern, 

    def predict_labels(channels: List[str], data: np.ndarray, fs: float, reference_system: str, model_name: str = 'model.json') -> Dict[str, Any]:

haben wir die folgenden Code in "predict_labels" verwendet, um unsere Modelle zu laden.

    try:
        gbdt_model_name = f"{model_name.replace('.json', '')}_gbdt.json"
        wbcknn_model_name = f"{model_name.replace('.json', '')}_wbcknn.json"

Daher haben wir bei der Abgabe diese Formulierung verwendet. Wenn Sie irgendwelche Fragen haben, stehen wir Ihnen jederzeit zur Verfügung.

    abgabe = {
        "team_name": "EEG-FV",
        "git_SSH": "git@github.com:Justus-wang-233/EEG-FV.git",
        "model": "model.json",
        "python_version": 3.8
             }
