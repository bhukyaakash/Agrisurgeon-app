document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('dropZone');
    const imageInput = document.getElementById('imageInput');
    const predictBtn = document.getElementById('predictBtn');
    const clearBtn = document.getElementById('clearBtn');
    const resultsSection = document.getElementById('resultsSection');
    const loadingSpinner = document.getElementById('loadingSpinner');

    let selectedImage = null;

    // Drag and drop
    dropZone.addEventListener('click', () => imageInput.click());
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#27ae60';
        dropZone.style.background = 'rgba(46, 204, 113, 0.05)';
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = '#2ecc71';
        dropZone.style.background = '#ecf0f1';
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#2ecc71';
        dropZone.style.background = '#ecf0f1';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            imageInput.files = files;
            handleImageSelect();
        }
    });

    imageInput.addEventListener('change', handleImageSelect);

    function handleImageSelect() {
        if (imageInput.files.length > 0) {
            selectedImage = imageInput.files[0];
            dropZone.innerHTML = `<p>✓ Image selected: ${selectedImage.name}</p>`;
            dropZone.style.borderColor = '#27ae60';
        }
    }

    predictBtn.addEventListener('click', async () => {
        if (!selectedImage) {
            alert('Please select an image first');
            return;
        }

        const temperature = document.getElementById('temperature').value;
        const humidity = document.getElementById('humidity').value;

        const formData = new FormData();
        formData.append('file', selectedImage);
        formData.append('temperature', temperature);
        formData.append('humidity', humidity);

        loadingSpinner.classList.remove('hidden');
        predictBtn.disabled = true;

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Prediction failed');
            }

            const data = await response.json();
            displayResults(data);

        } catch (error) {
            console.error('Error:', error);
            alert('Error processing image. Please try again.');
        } finally {
            loadingSpinner.classList.add('hidden');
            predictBtn.disabled = false;
        }
    });

    function displayResults(data) {
        document.getElementById('diseaseName').textContent = data.disease;
        document.getElementById('confidence').textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
        document.getElementById('conditionInfo').textContent = 
            `Temperature: ${data.temperature}°C, Humidity: ${data.humidity}%`;
        
        document.getElementById('cause').textContent = data.advisory.cause || 'N/A';
        document.getElementById('cure').textContent = data.advisory.cure || 'N/A';
        document.getElementById('prevention').textContent = data.advisory.prevention || 'N/A';

        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    clearBtn.addEventListener('click', () => {
        imageInput.value = '';
        selectedImage = null;
        dropZone.innerHTML = `
            <svg class="upload-icon" viewBox="0 0 24 24">
                <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/>
            </svg>
            <p>Drag and drop your image here or click to select</p>
        `;
        resultsSection.style.display = 'none';
    });
});