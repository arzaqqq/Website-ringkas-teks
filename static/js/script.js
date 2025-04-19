document.getElementById('summarize-form').addEventListener('submit', function() {
    const normalText = document.querySelector('.normal-text');
    const loadingSpinner = document.querySelector('.loading-spinner');
    
    normalText.style.display = 'none';
    loadingSpinner.style.display = 'inline-block';
});
