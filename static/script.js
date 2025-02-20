document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("eventForm").addEventListener("submit", function (event) {
        event.preventDefault();

        let formData = new FormData(this);
        fetch("/predict", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById("result").innerText = "Prediction: " + data.prediction;
        })
        .catch(error => console.error("Error:", error));
    });
});
