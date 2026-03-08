'use strict';

const currentPage = window.location.pathname.split('/').pop() || 'index.html';
        console.log('Current page:', currentPage);
        
        // Get all nav links
        const navLinks = document.querySelectorAll('.nav-links a');
        
        // Loop through links and add active class to matching page
        navLinks.forEach(link => {
            const linkPage = link.getAttribute('href');
            if (linkPage === currentPage) {
                link.classList.add('active');
                // Update page title
                document.getElementById('page-title').textContent = 
                    link.textContent + ' Page';
            }
        });
        
        // If no match found (e.g., different page name), default to first link
        if (!document.querySelector('.nav-links a.active')) {
            navLinks[0].classList.add('active');
        }