// script.js
document.getElementById('newsForm').addEventListener('submit', function(e) {
    e.preventDefault();

    const newsInput = document.getElementById('newsInput').value;
    const resultDiv = document.getElementById('result');

    if (newsInput.trim() === "") {
        resultDiv.innerHTML = "<p style='color: red;'>Please enter some news text.</p>";
        return;
    }

    // Simulate API call to backend for fake news detection
    // In a real-world app, this would be a POST request to a backend API
    resultDiv.innerHTML = "<p>Analyzing...</p>";

    setTimeout(() => {
        // Simulated result (in real case, you'd get the result from the API)
        const isFake = Math.random() > 0.5; // Random true/false for fake news

        if (isFake) {
            resultDiv.innerHTML = "<p style='color: red;'>This news seems to be fake!</p>";
        } else {
            resultDiv.innerHTML = "<p style='color: green;'>This news seems to be real!</p>";
        }
    }, 2000);
});
