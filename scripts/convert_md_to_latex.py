#!/usr/bin/env python3
"""
Convert Intrinsic-Resonance-Holography-21.4.md to LaTeX format.

This script:
1. Removes the prologue section
2. Extracts and numerically references all citations
3. Converts markdown to LaTeX with proper formatting
4. Sets up Noto Sans (text) and Fira Math (mathematics) fonts
5. Creates colored hyperlinks (citations: green, URLs: magenta, internal: blue)
6. Adds figure placeholders with detailed descriptions
"""

import re
import os
from pathlib import Path
from typing import List, Dict, Tuple

class MarkdownToLaTeXConverter:
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        self.citations = []  # List of (author, year, title, url/doi) tuples
        self.citation_map = {}  # Map citation text to citation number
        self.figure_count = 0
        self.table_count = 0
        
    def extract_citations(self, content: str) -> None:
        """Extract all citations from the document."""
        # Common author names and years
        patterns = [
            r'(Reuter|Percacci|Wetterich|Rovelli|Smolin|Weinberg|Penrose|Hawking|Wheeler|Bekenstein|Hooft|Susskind|Maldacena|Witten|Polchinski|Litim|Oriti|Gielen|Lloyd|Fredkin|Wolfram)\s*(\d{4}|\(?\d{4}\)?)',
            r'(Wheeler-DeWitt|Bekenstein-Hawking)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                citation_text = match.group(0)
                if citation_text not in self.citation_map:
                    self.citation_map[citation_text] = len(self.citations) + 1
                    self.citations.append(citation_text)
    
    def convert_math(self, text: str) -> str:
        """Convert markdown math delimiters to LaTeX."""
        # Inline math: $...$ stays the same
        # Display math: ```math ... ``` -> \[ ... \]
        text = re.sub(r'```math\n(.*?)\n```', r'\\[\n\\1\n\\]', text, flags=re.DOTALL)
        return text
    
    def convert_headers(self, text: str) -> str:
        """Convert markdown headers to LaTeX sections."""
        # # Header -> \section{Header}
        # ## Header -> \subsection{Header}
        # ### Header -> \subsubsection{Header}
        # #### Header -> \paragraph{Header}
        text = re.sub(r'^#### (.*?)$', r'\\paragraph{\1}', text, flags=re.MULTILINE)
        text = re.sub(r'^### (.*?)$', r'\\subsubsection{\1}', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.*?)$', r'\\subsection{\1}', text, flags=re.MULTILINE)
        text = re.sub(r'^# (.*?)$', r'\\section{\1}', text, flags=re.MULTILINE)
        return text
    
    def convert_emphasis(self, text: str) -> str:
        """Convert markdown emphasis to LaTeX."""
        # **bold** -> \textbf{bold}
        text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)
        # *italic* -> \textit{italic}
        text = re.sub(r'\*(.+?)\*', r'\\textit{\1}', text)
        return text
    
    def convert_lists(self, text: str) -> str:
        """Convert markdown lists to LaTeX."""
        # This is a simplified version - would need more sophisticated handling
        lines = text.split('\n')
        result = []
        in_list = False
        
        for line in lines:
            if re.match(r'^[\*\-]\s+', line):
                if not in_list:
                    result.append('\\begin{itemize}')
                    in_list = True
                item = re.sub(r'^[\*\-]\s+', '', line)
                result.append(f'\\item {item}')
            else:
                if in_list:
                    result.append('\\end{itemize}')
                    in_list = False
                result.append(line)
        
        if in_list:
            result.append('\\end{itemize}')
        
        return '\n'.join(result)
    
    def add_citation_refs(self, text: str) -> str:
        """Add citation references to the text."""
        for citation_text, cite_num in self.citation_map.items():
            # Add \cite{} markers
            text = text.replace(citation_text, f'{citation_text}\\cite{{ref{cite_num}}}')
        return text
    
    def create_latex_preamble(self) -> str:
        """Create the LaTeX document preamble."""
        return r"""\documentclass[12pt,a4paper]{article}

% Font packages
\usepackage{fontspec}
\usepackage{unicode-math}

% Set fonts
\setmainfont{Noto Sans}
\setmathfont{Fira Math}

% Mathematics packages
\usepackage{amsmath}
\usepackage{amsthm}
\usepackage{amssymb}

% Graphics and figures
\usepackage{graphicx}
\usepackage{float}

% Hyperlinks with colors
\usepackage{xcolor}
\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=blue,      % Internal links in blue
    citecolor=green,     % Citations in green
    urlcolor=magenta,    % URLs in magenta
    bookmarksnumbered=true,
    pdfstartview={FitH},
    breaklinks=true
}

% Page layout
\usepackage[margin=1in]{geometry}

% Theorem environments
\newtheorem{theorem}{Theorem}[section]
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{definition}[theorem]{Definition}

% Figure placeholder command
\newcommand{\figureplaceholder}[2]{%
    \begin{figure}[H]
        \centering
        \fbox{\parbox{0.9\textwidth}{
            \textbf{[FIGURE PLACEHOLDER]}\\[0.5em]
            \textit{Figure #1: #2}
        }}
    \end{figure}
}

\begin{document}
"""
    
    def create_title_page(self) -> str:
        """Create the title page."""
        return r"""
\title{Intrinsic Resonance Holography v21.4:\\
The Architectonic Rectification of Quantum Mechanics,\\
General Relativity, and the Standard Model from a\\
Quantum-Informational Field Theory}

\author{Brandon D. McCrary\\
\textit{Independent Theoretical Physics Researcher}\\
\href{https://orcid.org/0009-0008-2804-7165}{ORCID: 0009-0008-2804-7165}}

\date{December 2025 (v21.4)}

\maketitle

\begin{abstract}
Intrinsic Resonance Holography (IRH) presents a comprehensive theoretical framework that unifies Quantum Mechanics, General Relativity, and the Standard Model of particle physics. It posits that reality fundamentally emerges from a Cymatic Group Field Theory (cGFT) defined on a primordial quantum-informational substrate, the group manifold $G_{\text{inf}} = \mathrm{SU}(2) \times \mathrm{U}(1)$. The universe's observed laws and constants are shown to be the unique, mathematically inevitable consequence of this cGFT undergoing an asymptotically safe renormalization group (RG) flow towards a Cosmic Fixed Point.
\end{abstract}

\tableofcontents
\newpage
"""
    
    def create_bibliography(self) -> str:
        """Create the bibliography section."""
        bib_entries = []
        
        # Define known references
        known_refs = {
            'Reuter 1998': ('M. Reuter', '1998', 'Nonperturbative evolution equation for quantum gravity', 'Phys. Rev. D 57, 971'),
            'Percacci 2017': ('R. Percacci', '2017', 'An Introduction to Covariant Quantum Gravity and Asymptotic Safety', 'World Scientific'),
            'Wetterich': ('C. Wetterich', '1993', 'Exact evolution equation for the effective potential', 'Phys. Lett. B 301, 90'),
            'Wheeler': ('J. A. Wheeler', '1990', 'Information, physics, quantum: The search for links', 'Proc. 3rd Int. Symp. Foundations of Quantum Mechanics'),
            'Wheeler-DeWitt': ('B. S. DeWitt', '1967', 'Quantum Theory of Gravity. I. The Canonical Theory', 'Phys. Rev. 160, 1113'),
            'Bekenstein-Hawking': ('J. D. Bekenstein', '1973', 'Black Holes and Entropy', 'Phys. Rev. D 7, 2333'),
            'Susskind': ('L. Susskind', '1995', 'The World as a Hologram', 'J. Math. Phys. 36, 6377'),
            'Weinberg': ('S. Weinberg', '1979', 'Ultraviolet divergences in quantum theories of gravitation', 'in General Relativity: An Einstein Centenary Survey'),
        }
        
        bib = r"""
\begin{thebibliography}{99}
"""
        
        for i, citation_text in enumerate(self.citations, 1):
            # Try to find in known refs
            ref_found = False
            for key, (author, year, title, journal) in known_refs.items():
                if key.lower() in citation_text.lower():
                    bib += f"\\bibitem{{ref{i}}} {author} ({year}). \\textit{{{title}}}. {journal}.\n\n"
                    ref_found = True
                    break
            
            if not ref_found:
                bib += f"\\bibitem{{ref{i}}} {citation_text}. [To be completed with full citation details]\n\n"
        
        # Add GitHub repository
        bib += r"""\bibitem{github} IRH Repository. \url{https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonace_Holography-.git}

\bibitem{orcid} Brandon D. McCrary. ORCID Profile. \url{https://orcid.org/0009-0008-2804-7165}

\end{thebibliography}
"""
        return bib
    
    def process_figure_placeholders(self, text: str) -> str:
        """Add detailed figure placeholders."""
        # Find figure references
        figure_pattern = r'#### Figure (\d+\.\d+): (.+?)$'
        
        def replace_figure(match):
            self.figure_count += 1
            fig_num = match.group(1)
            caption = match.group(2)
            
            # Try to find description after the figure reference
            return f"\\figureplaceholder{{{fig_num}}}{{{caption}}}\n"
        
        text = re.sub(figure_pattern, replace_figure, text, flags=re.MULTILINE)
        return text
    
    def skip_prologue(self, content: str) -> str:
        """Skip the prologue and start from section 1."""
        # Find the start of section 1
        match = re.search(r'^## 1\. Formal Foundation', content, re.MULTILINE)
        if match:
            return content[match.start():]
        return content
    
    def convert(self):
        """Main conversion method."""
        print(f"Reading {self.input_file}...")
        with open(self.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("Extracting citations...")
        self.extract_citations(content)
        print(f"Found {len(self.citations)} unique citations")
        
        print("Skipping prologue...")
        content = self.skip_prologue(content)
        
        print("Converting markdown to LaTeX...")
        content = self.convert_math(content)
        content = self.convert_headers(content)
        content = self.convert_emphasis(content)
        content = self.convert_lists(content)
        content = self.process_figure_placeholders(content)
        content = self.add_citation_refs(content)
        
        print(f"Writing LaTeX to {self.output_file}...")
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(self.create_latex_preamble())
            f.write(self.create_title_page())
            f.write(content)
            f.write("\n\\newpage\n")
            f.write(self.create_bibliography())
            f.write("\n\\end{document}\n")
        
        print(f"Conversion complete! LaTeX file created: {self.output_file}")
        print(f"Total citations: {len(self.citations)}")
        print(f"Total figures: {self.figure_count}")

def main():
    input_file = "/home/runner/work/Intrinsic_Resonance_Holography/Intrinsic_Resonance_Holography/Intrinsic-Resonance-Holography-21.4.md"
    output_file = "/home/runner/work/Intrinsic_Resonance_Holography/Intrinsic_Resonance_Holography/Intrinsic-Resonance-Holography-21.4.tex"
    
    converter = MarkdownToLaTeXConverter(input_file, output_file)
    converter.convert()

if __name__ == "__main__":
    main()
