document.addEventListener("DOMContentLoaded", () => {
    const toggle = document.getElementById("menu-toggle");
    const navbar = document.getElementById("navbar");
    const navLinks = navbar.querySelectorAll("a");
  
    toggle.addEventListener("click", () => {
      navbar.classList.toggle("active");
    });
  
    navLinks.forEach(link => {
      link.addEventListener("click", () => {
        navbar.classList.remove("active");
      });
    });
  });
  


let currentSlide = 0;

function showSlide(index) {
    const slides = document.querySelectorAll('.text-slide');
    slides.forEach((slide, i) => {
        slide.classList.remove('active');
        if (i === index) {
            slide.classList.add('active');
        }
    });
}

function changeSlide(direction) {
    const slides = document.querySelectorAll('.text-slide');
    currentSlide = (currentSlide + direction + slides.length) % slides.length;
    showSlide(currentSlide);
}

document.addEventListener('DOMContentLoaded', () => {
    showSlide(currentSlide);
    setInterval(() => changeSlide(1), 4000);
});


  


