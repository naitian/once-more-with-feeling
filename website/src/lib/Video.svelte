<script>
	import Wide from './Wide.svelte';
	import { intersect } from 'svelte-intersection-observer-action';

	const intersectionOptions = {
		callback: (entry) => {
			if (entry.isIntersecting) {
				entry.target.play();
			} else {
				entry.target.pause();
			}
		},
		threshold: [0.75]
	};
</script>

<Wide>
	<div id="theater" class="video-container">
		<div class="inner">
			<video src="/ladies_and_gentlemen.mp4" use:intersect={intersectionOptions} controls></video>
            <p class="caption">Keep scrolling to dive in.</p>
		</div>
	</div>
</Wide>

<style>
	.video-container {
		min-height: 100vh;
		position: relative;
	}
	.inner {
        width: 100%;
		position: absolute;
		top: 50%;
		transform: translateY(-50%);
	}
	video {
		width: 100%;
	}

    .caption {
        text-align: center;
        opacity: 0.8;
    }
</style>
