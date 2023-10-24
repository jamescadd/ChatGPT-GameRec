class Video(object):

    def __init__(self, url: str, captions: [str]):
        self.url = url
        self.captions = captions

    def get_transcription(self, preserve_lines: bool = False) -> str:
        """
        Return the video captions as a string

        Parameters
        ----------
        preserve_lines (bool): keep line separators from original transcript extraction

        Returns
        -------
        str: the transcript as a single string
        """
        if preserve_lines:
            return '\n'.join(self.captions)
        return ' '.join(self.captions)
