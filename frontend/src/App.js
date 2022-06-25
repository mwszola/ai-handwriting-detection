import {useState} from 'react';
import './App.css';



function App() {
    const [response, setResponse] = useState("Proszę przesłać zdjęcie");

    const handleChange = (event) => {
        setResponse("Identyfikowanie tekstu");
        const {target: {files}} = event;
        const image = files[0];

        async function predictResults() {
            const data = new FormData();
            data.append("image", image)
            try {
                const response = await fetch("http://localhost:5000/predict", {
                    method: "post",
                    body: data
                });
                const result = await response.json();
                const {prediction} = result;
                setResponse(`Myślę, że ta litera to ${prediction}`);
            } catch (err) {
                setResponse("Wystąpił błąd");
            }
        }

        predictResults();
    }

    return (
        <div className="App">
            <header className="App-header">
                <h1>Rozpoznawanie znaków pisanych</h1>
                <input
                    className="App-fileInput"
                    type="file"
                    onChange={handleChange}
                />
                <p>{response}</p>
            </header>
        </div>
    );
}

export default App;
