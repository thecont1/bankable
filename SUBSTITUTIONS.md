# Text Substitutions Config
#
# Applied to the .md content BEFORE sending to the LLM for transaction extraction.
# Each active line defines a find → replace pair using quoted strings:
#
#   "find_text" → "replace_text"
#
# Lines starting with # are comments. Blank lines are ignored.
# Substitutions are applied in order, top to bottom.

# Clean up HTML line breaks from Marker output
"<br>" → " "
"*" → " "

# Fix common OCR concatenation errors (merchant + city jammed together)
"SMARTBUYBANGALORE" → "SMARTBUY BANGALORE"
"BLINKITGURGAON" → "BLINKIT GURGAON"
"YOUTUBEGOOGLE" → "YOUTUBE GOOGLE"
"GOOGLEPLAY" → "GOOGLE PLAY"
"TRADESBENGALURU" → "TRADES BENGALURU"

# Normalize common merchant names
"TECHNOLOGIESBENGALURU" → "TECHNOLOGIES BENGALURU"
