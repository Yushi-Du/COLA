window.HELP_IMPROVE_VIDEOJS = false;

document.addEventListener('DOMContentLoaded', function () {
    const loader = document.querySelector('#homepage-loader');
    const status = document.querySelector('#homepage-loader-status');
    const progress = document.querySelector('#homepage-loader-progress');
    const retry = document.querySelector('#homepage-loader-retry');
    const videos = Array.from(document.querySelectorAll('video'));
    let generation = 0;

    const revealTargets = Array.from(document.querySelectorAll([
        '.head-section .has-text-centered',
        '.content-section > .container > .columns',
        '.content-section .columns.is-multiline > .column',
        '.footer .column'
    ].join(', ')));
    revealTargets.forEach(function (element, index) {
        element.classList.add('project-reveal');
        element.style.setProperty('--project-reveal-delay', `${(index % 3) * 65}ms`);
    });

    function startRevealAnimations() {
        document.body.classList.add('page-ready');
        if (!('IntersectionObserver' in window)) {
            revealTargets.forEach(function (element) { element.classList.add('is-visible'); });
            return;
        }
        const observer = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (!entry.isIntersecting) return;
                entry.target.classList.add('is-visible');
                observer.unobserve(entry.target);
            });
        }, { threshold: 0.12, rootMargin: '0px 0px -7% 0px' });
        revealTargets.forEach(function (element) { observer.observe(element); });
    }

    if (!loader || videos.length === 0) {
        document.body.classList.remove('page-loading');
        return;
    }

    function updateProgress(readyCount) {
        status.textContent = `Loading contents ${readyCount} / ${videos.length}`;
        progress.style.width = `${100 * readyCount / videos.length}%`;
    }

    function waitUntilVisible(video) {
        if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
            return Promise.resolve();
        }
        return new Promise(function (resolve, reject) {
            function cleanUp() {
                video.removeEventListener('loadeddata', onReady);
                video.removeEventListener('error', onError);
            }
            function onReady() {
                cleanUp();
                resolve();
            }
            function onError() {
                cleanUp();
                reject(new Error('A homepage video could not be loaded'));
            }
            video.addEventListener('loadeddata', onReady);
            video.addEventListener('error', onError);
        });
    }

    function revealHomepage() {
        videos.forEach(function (video) {
            const playRequest = video.play();
            if (playRequest) playRequest.catch(function () {});
        });
        requestAnimationFrame(function () {
            document.body.classList.remove('page-loading');
            loader.classList.add('is-complete');
            loader.setAttribute('aria-hidden', 'true');
            startRevealAnimations();
        });
    }

    function loadHomepageVideos(forceReload) {
        const currentGeneration = ++generation;
        let readyCount = 0;
        loader.classList.remove('is-error');
        loader.removeAttribute('aria-hidden');
        retry.hidden = true;
        updateProgress(0);

        if (forceReload) videos.forEach(function (video) { video.load(); });

        Promise.all(videos.map(function (video) {
            return waitUntilVisible(video).then(function () {
                readyCount += 1;
                if (currentGeneration === generation) updateProgress(readyCount);
            });
        })).then(function () {
            if (currentGeneration === generation) revealHomepage();
        }).catch(function () {
            if (currentGeneration !== generation) return;
            loader.classList.add('is-error');
            status.textContent = `Content loading stopped at ${readyCount} / ${videos.length}`;
            retry.hidden = false;
        });
    }

    retry.addEventListener('click', function () { loadHomepageVideos(true); });
    loadHomepageVideos(false);
});

class BeforeAfter {
	constructor(enteryObject) {

			const beforeAfterContainer = document.querySelector(enteryObject.id);
			const before = beforeAfterContainer.querySelector('.bal-before');
			const handle = beforeAfterContainer.querySelector('.bal-handle');
			var widthChange = 0;

			beforeAfterContainer.querySelector('.bal-before-inset').setAttribute("style", "width: " + beforeAfterContainer.offsetWidth + "px;")
			window.onresize = function () {
					beforeAfterContainer.querySelector('.bal-before-inset').setAttribute("style", "width: " + beforeAfterContainer.offsetWidth + "px;")
			}
			before.setAttribute('style', "width: 50%;");
			handle.setAttribute('style', "left: 50%;");

			//mouse move event listener
			beforeAfterContainer.addEventListener('mousemove', (e) => {
					let containerWidth = beforeAfterContainer.offsetWidth;
					widthChange = e.offsetX;
					let newWidth = widthChange * 100 / containerWidth;

					if (e.offsetX > 20 && e.offsetX < beforeAfterContainer.offsetWidth - 20) {
							before.setAttribute('style', "width:" + newWidth + "%;");
							handle.setAttribute('style', "left:" + newWidth + "%;");
					}
			})
	}
}

$(document).ready(function() {
    // Check for click events on the navbar burger icon
    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: false,
			autoplay: false,
			autoplaySpeed: 100000,
    }
		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
    bulmaSlider.attach();
})
