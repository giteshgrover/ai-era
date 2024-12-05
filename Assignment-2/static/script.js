// const checkboxes = document.querySelectorAll('input[name="animal"]');
//     const animalImage = document.getElementById('animalImage');
    
//     // Create image elements for each animal
//     const images = {
//         cat: createImage('static/images/cat.jpg'),
//         dog: createImage('static/images/dog.jpg'),
//         elephant: createImage('static/images/elephant.jpg')
//     };
    
//     // Add all images to the container
//     Object.values(images).forEach(img => animalImage.appendChild(img));
    
    function createImage(src) {
        const img = document.createElement('img');
        img.src = src;
        img.style.display = 'none';
        return img;
    }
    
//     checkboxes.forEach(checkbox => {
//         checkbox.addEventListener('change', checkBoxChange);
//     });

//     function checkBoxChange(e) {
//         (e) => {
//             if (e.target.checked) {
//                 // Uncheck other checkboxes
//                 checkboxes.forEach(cb => {
//                     if (cb !== e.target) cb.checked = false;
//                 });
                
//                 // Hide all images
//                 Object.values(images).forEach(img => img.style.display = 'none');
                
//                 // Show selected animal's image
//                 const selectedAnimal = e.target.value;
//                 if (images[selectedAnimal]) {
//                     images[selectedAnimal].style.display = 'inline-block';
//                 }
//             } else {
//                 // Hide the image when unchecking
//                 images[e.target.value].style.display = 'none';
//             }
//     }


    const images = {};
    const checkboxes = [];
    // First, fetch the animals data when the page loads
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch('/animals');
        if (!response.ok) throw new Error('Failed to fetch animals');
        const data = await response.json();
        console.log('Animals data:', data); // Debug log
        const checkboxGroup = document.getElementById('checkbox-group'); 
        const animalImage = document.getElementById('animalImage');   
        data.animals.forEach(animal => {
            const label = document.createElement('label');
            label.textContent = animal.name;
            const input = createInputElement(animal.name);
            label.appendChild(input);
            checkboxes.push(input);
            checkboxGroup.appendChild(label);

            // Add all images to the container
            const img = createImage(animal.image);
            animalImage.appendChild(img);
            images[animal.name] = img;
        });

        checkboxes.forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                if (e.target.checked) {
                    // Uncheck other checkboxes
                    checkboxes.forEach(cb => {
                        if (cb !== e.target) cb.checked = false;
                    });
                    
                    // Hide all images
                    Object.values(images).forEach(img => img.style.display = 'none');
                    
                    // Show selected animal's image
                    const selectedAnimal = e.target.value;
                    if (images[selectedAnimal]) {
                        images[selectedAnimal].style.display = 'inline-block';
                    }
                } else {
                    // Hide the image when unchecking
                    images[e.target.value].style.display = 'none';
                }
            });
        });

    } catch (error) {
        console.error('Error fetching animals:', error);
    }
});

function createInputElement(name) { 
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.name = 'animal';
    input.value = name;
    return input;
}

// Handle file upload
const fileInput = document.getElementById('fileInput');
console.log('fileInput');
console.log(fileInput);
fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    console.log("Here")
    if (file) {
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/upload/', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Upload failed');
            }

            const data = await response.json();
            console.log(data);
            
            // Create or update file details display
            const fileDetails = document.getElementById('fileDetails') || createFileDetailsElement();
            fileDetails.innerHTML = `
                <p><strong>File Name:</strong> ${data.filename}</p>
                <p><strong>File Type:</strong> ${data.content_type}</p>
                <p><strong>File Size:</strong> ${formatFileSize(data.file_size)}</p>
                <p class="success-message">File uploaded successfully!</p>
            `;
        } catch (error) {
            console.error('Error:', error);
            const fileDetails = document.getElementById('fileDetails') || createFileDetailsElement();
            fileDetails.innerHTML = `
                <p class="error-message">Error uploading file: ${error.message}</p>
            `;
        }
    }
});

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function createFileDetailsElement() {
    const fileDetails = document.createElement('div');
    fileDetails.id = 'fileDetails';
    fileDetails.className = 'file-details';
    const fileUploadDiv = document.querySelector('.file-upload');
    fileUploadDiv.appendChild(fileDetails);
    return fileDetails;
}