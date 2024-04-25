function processImage() {
    const formData = new FormData(document.getElementById('imageForm'));

    // Send the image file to the backend for processing
    fetch('/process', {
        method: 'POST',
        body: formData
    })
    .then(response => response.blob())
    .then(blob => {
        const resultUrl = URL.createObjectURL(blob);
        document.getElementById('result').innerHTML = `<img src="${resultUrl}" alt="Cartoonified Image">`;
    })
    .catch(error => console.error('Error:', error));
}
