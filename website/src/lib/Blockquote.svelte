<script>
	export let author = '';
    export let source = '';
</script>

<div class="blockquote-container">
	<blockquote class="blockquote">
		<p>
			“<slot></slot>”
		</p>
		<footer>— {author} <br><i>{source}</i></footer>
	</blockquote>
</div>

<style>
	.blockquote {
		box-sizing: border-box;
		max-width: 80%;
		font-family: 'Georgia', serif;
		font-size: 1.2em;
		line-height: 1.6;
		position: relative;
		padding: 20px;
	}

	.blockquote p {
		margin: 0;
        color: #aaa;
	}

	.blockquote footer {
		margin-top: 10px;
		font-size: 0.8em;
		color: #999;
		text-align: right;
	}

	.blockquote::before {
		content: open-quote;
		font-size: 4em;
		opacity: 0.1;
		position: absolute;
		top: -10px;
		left: -10px;
	}

	.blockquote::after {
		content: close-quote;
		font-size: 4em;
		opacity: 0.1;
		position: absolute;
		bottom: -10px;
		right: -10px;
	}
</style>
