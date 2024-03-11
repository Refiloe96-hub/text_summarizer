function handleSuccessResponse(response) {
    // Check if the response contains the summary and topics
    if (response.summary && response.topics) {
        // Display the summary and topics on your webpage
        $('#summaryArea').text(response.summary); 
        let topicsList = '';
        response.topics.forEach(topic => {
            topicsList += `<li>${topic}</li>`;
        });
        $('#topicsArea').html(topicsList);
    } else if (response.error) {
        // Handle any errors sent from the server
        toastr.error(response.error);
    }
}

function uploadVideo(formData) {
    $.ajax({
        type: "POST",
        url: "/video-upload",
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
            handleSuccessResponse(response);
        },
        error: function(error) {
            toastr.error("Error uploading video: " + error.responseText);
        }
    });
}

toastr.options = {
    "closeButton": true,
    "debug": false,
    "newestOnTop": false,
    "progressBar": false,
    "positionClass": "toast-top-right",
    "preventDuplicates": false,
    "onclick": null,
    "showDuration": "300",
    "hideDuration": "1000",
    "timeOut": "3000",
    "extendedTimeOut": "1000",
    "showEasing": "swing",
    "hideEasing": "linear",
    "showMethod": "fadeIn",
    "hideMethod": "fadeOut"
};

document.addEventListener('DOMContentLoaded', function() {
    // Copy Icon Logic
    var copyIcon = document.getElementById('copyIcon');
    var summaryText = document.getElementById('summaryText');
    if (copyIcon && summaryText) {
        copyIcon.addEventListener('click', function() {
            var textArea = document.createElement("textarea");
            textArea.value = summaryText.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand("copy");
            document.body.removeChild(textArea);
            toastr.success('Summary copied to clipboard!');
        });
    }

    // Theme Toggle Logic
    const themeToggle = document.getElementById('themeToggle');
    const currentTheme = localStorage.getItem('theme') || "light";
    document.documentElement.setAttribute('data-theme', currentTheme);
    themeToggle.innerText = currentTheme === 'dark' ? "Light Mode" : "Dark Mode";

    themeToggle.addEventListener('click', function() {
        if (document.documentElement.getAttribute('data-theme') === 'dark') {
            document.documentElement.setAttribute('data-theme', 'light');
            localStorage.setItem('theme', 'light');
            themeToggle.innerText = "Dark Mode";
        } else {
            document.documentElement.setAttribute('data-theme', 'dark');
            localStorage.setItem('theme', 'dark');
            themeToggle.innerText = "Light Mode";
        }
    });
});
