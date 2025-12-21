# LaTeX Compilation Instructions for IRH v21.4

## Overview
The Intrinsic-Resonance-Holography-21.4.tex file has been converted from the original Markdown manuscript with the following specifications:

- **Text Font**: Noto Sans
- **Math Font**: Fira Math  
- **Citation Links**: Green color (#008000)
- **URL Links**: Magenta color (#FF00FF)
- **Internal Links**: Blue color (#0000FF)
- **Prologue**: Removed (document starts from Section 1)
- **Citations**: Numerically referenced with embedded bibliography
- **Figures**: Placeholders with detailed descriptions

## Requirements

### LaTeX Distribution
You need a modern LaTeX distribution with XeLaTeX or LuaLaTeX support:
- **Linux**: TeX Live 2020 or later
- **macOS**: MacTeX 2020 or later
- **Windows**: MiKTeX 2.9 or later

### Required Fonts
The document uses system fonts that must be installed:

1. **Noto Sans**: Download from [Google Fonts](https://fonts.google.com/noto/specimen/Noto+Sans)
2. **Fira Math**: Download from [CTAN](https://www.ctan.org/pkg/fira) or [GitHub](https://github.com/firamath/firamath)

#### Installing Fonts

**Linux (Ubuntu/Debian)**:
```bash
sudo apt-get install fonts-noto
# For Fira Math, download and install manually or use:
wget https://github.com/firamath/firamath/releases/download/v0.3.4/FiraMath-Regular.otf
sudo mkdir -p /usr/local/share/fonts/fira
sudo cp FiraMath-Regular.otf /usr/local/share/fonts/fira/
sudo fc-cache -fv
```

**macOS**:
```bash
brew tap homebrew/cask-fonts
brew install font-noto-sans
# Download Fira Math manually and install via Font Book
```

**Windows**:
- Download font files
- Right-click and select "Install" or copy to `C:\Windows\Fonts\`

## Compilation

### Using XeLaTeX (Recommended)
```bash
cd /home/runner/work/Intrinsic_Resonance_Holography/Intrinsic_Resonance_Holography
xelatex Intrinsic-Resonance-Holography-21.4.tex
xelatex Intrinsic-Resonance-Holography-21.4.tex  # Run twice for references
```

### Using LuaLaTeX (Alternative)
```bash
lualatex Intrinsic-Resonance-Holography-21.4.tex
lualatex Intrinsic-Resonance-Holography-21.4.tex  # Run twice for references
```

### Why Two Passes?
Running the LaTeX compiler twice ensures:
1. First pass: Generates auxiliary files (.aux, .toc)
2. Second pass: Resolves all cross-references and citations

## Conversion Script

The conversion was performed using the enhanced Python script:
```bash
python scripts/enhanced_latex_converter.py
```

### Script Features
- **Citation Extraction**: Automatically identifies and numbers 12+ references
- **Math Environment Conversion**: Converts markdown math blocks to LaTeX equations
- **Section Hierarchy**: Preserves document structure with proper LaTeX sections
- **Figure Placeholders**: Creates detailed placeholder boxes for figures
- **Hyperlink Coloring**: Implements the specified color scheme for links

## Document Structure

```
Intrinsic-Resonance-Holography-21.4.tex
├── Preamble (fonts, packages, colors)
├── Title Page
├── Abstract
├── Table of Contents
├── Main Content (Sections 1-15)
│   ├── Section 1: Formal Foundation
│   ├── Section 2: Emergence of Spacetime  
│   ├── Section 3: Emergence of Standard Model
│   ├── ...
│   └── Section 15: Ethical Approval
├── Appendices (A-K)
└── References (Bibliography)
```

## Customization

### Changing Colors
Edit the preamble color definitions:
```latex
\definecolor{citecolor}{RGB}{0,128,0}     % Green for citations
\definecolor{urlcolor}{RGB}{255,0,255}    % Magenta for URLs  
\definecolor{linkcolor}{RGB}{0,0,255}     % Blue for internal links
```

### Changing Fonts
Modify the font specifications:
```latex
\setmainfont{Noto Sans}
\setmathfont{Fira Math}
```

### Page Layout
Adjust margins and spacing:
```latex
\usepackage[margin=1in]{geometry}
\onehalfspacing  % Change to \doublespacing for double spacing
```

## Troubleshooting

### Font Not Found Error
```
! fontspec error: "font-not-found"
! The font "Noto Sans" cannot be found.
```
**Solution**: Install the required fonts (see Requirements section)

### Math Font Issues
```
! Package unicode-math Error: The math font "Fira Math" does not appear to be a
math font.
```
**Solution**: Ensure Fira Math is properly installed. Try using Fira Math Regular.otf specifically.

### Compilation Errors
If you encounter errors:
1. Check that you're using XeLaTeX or LuaLaTeX (not pdfLaTeX)
2. Ensure all required packages are installed
3. Update your TeX distribution: `tlmgr update --all`

### Alternative Fonts
If Noto Sans or Fira Math are unavailable, you can substitute:
```latex
\setmainfont{Arial}  % or Helvetica, or any sans-serif font
\setmathfont{XITS Math}  % or Latin Modern Math
```

## Output

The compiled PDF will contain:
- Professional typesetting with proper spacing and layout
- Colored hyperlinks for easy navigation
- Proper mathematical notation using Fira Math
- Clean, modern appearance with Noto Sans text
- Detailed figure placeholders indicating what visualizations should be added
- Complete bibliography with all referenced works

## Next Steps

### Adding Figures
The document contains placeholders marked as:
```
[FIGURE X.X PLACEHOLDER]
Caption: ...
Description: ...
```

To add actual figures:
1. Create figures based on the detailed descriptions
2. Replace placeholder boxes with:
```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/figure_X_X.pdf}
    \caption{...}
    \label{fig:X_X}
\end{figure}
```

### Citations
To add inline citations in the text, insert `\cite{refN}` where N is the reference number.

## Contact

For issues related to the conversion process or LaTeX compilation:
- Open an issue on the GitHub repository
- Check the IRH documentation in `docs/`

## License

This document and conversion scripts are part of the IRH project and follow the same license (Apache 2.0).
