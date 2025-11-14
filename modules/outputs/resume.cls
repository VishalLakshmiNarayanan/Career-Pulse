%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% resume.cls - Custom Resume Document Class
% Based on ASU Sparky Sundevil Resume Template
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{resume}[2024/01/01 Custom Resume Class]

% Load base class
\LoadClass[11pt,letterpaper]{article}

% Packages
\RequirePackage[parfill]{parskip}
\RequirePackage{array}
\RequirePackage{ifthen}
\RequirePackage[hidelinks]{hyperref}
\RequirePackage{enumitem}
\RequirePackage{geometry}
\RequirePackage{fancyhdr}
\RequirePackage{xcolor}
\RequirePackage{etoolbox}
\RequirePackage{titlesec}

% Page setup
\geometry{
    letterpaper,
    left=0.75in,
    right=0.75in,
    top=0.75in,
    bottom=0.75in
}

\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\setlength{\parindent}{0pt}

% Define colors
\definecolor{primary}{RGB}{0,0,0}
\definecolor{secondary}{RGB}{64,64,64}

% Section formatting
\titleformat{\section}
  {\Large\bfseries\scshape\raggedright}
  {}{0em}
  {}
  [\titlerule]

\titlespacing{\section}{0pt}{10pt}{5pt}

% ---------- Contact Information Command ----------
\newcommand{\introduction}[1]{
    \setkeys{contact}{#1}
    \begin{center}
        {\Huge\bfseries \contactfullname} \\[5pt]
        \contactemail \quad | \quad \contactphone \\
        \href{\contactlinkedin}{\contactlinkedin} \quad | \quad \href{\contactgithub}{\contactgithub}
    \end{center}
    \vspace{10pt}
}

\define@key{contact}{fullname}{\def\contactfullname{#1}}
\define@key{contact}{email}{\def\contactemail{#1}}
\define@key{contact}{phone}{\def\contactphone{#1}}
\define@key{contact}{linkedin}{\def\contactlinkedin{#1}}
\define@key{contact}{github}{\def\contactgithub{#1}}

% ---------- Summary Command ----------
\newcommand{\summary}[1]{
    \section*{Summary}
    #1
}

% ---------- Education Section ----------
\newenvironment{educationSection}[1]{
    \section*{#1}
}{}

\newcommand{\educationItem}[1]{
    \setkeys{education}{#1}
    \noindent
    \textbf{\educationuniversity} \hfill \educationgraduation \\
    \textit{\educationprogram} \hfill GPA: \educationgrade \\
    \ifthenelse{\equal{\educationcollege}{}}{}{%
        \educationcollege \\
    }
    \ifthenelse{\equal{\educationcoursework}{}}{}{%
        \textbf{Relevant Coursework:} \educationcoursework
    }
    \vspace{5pt}
}

\define@key{education}{university}{\def\educationuniversity{#1}}
\define@key{education}{college}{\def\educationcollege{#1}}
\define@key{education}{graduation}{\def\educationgraduation{#1}}
\define@key{education}{grade}{\def\educationgrade{#1}}
\define@key{education}{program}{\def\educationprogram{#1}}
\define@key{education}{coursework}{\def\educationcoursework{#1}}

% Set defaults
\setkeys{education}{college={}, coursework={}}

% ---------- Skills Section ----------
\newenvironment{skillsSection}[1]{
    \section*{#1}
    \begin{itemize}[leftmargin=*, itemsep=2pt, parsep=0pt]
}{\end{itemize}}

\newcommand{\skillItem}[1]{
    \setkeys{skill}{#1}
    \item \textbf{\skillcategory:} \skillskills
}

\define@key{skill}{category}{\def\skillcategory{#1}}
\define@key{skill}{skills}{\def\skillskills{#1}}

% ---------- Experience Section ----------
\newenvironment{experienceSection}[1]{
    \section*{#1}
}{}

\newcommand{\experienceItem}[1]{
    \setkeys{experience}{#1}
    \noindent
    \textbf{\experienceposition} \hfill \experienceduration \\
    \textit{\experiencecompany}, \experiencelocation
    \vspace{2pt}
}

\define@key{experience}{company}{\def\experiencecompany{#1}}
\define@key{experience}{location}{\def\experiencelocation{#1}}
\define@key{experience}{position}{\def\experienceposition{#1}}
\define@key{experience}{duration}{\def\experienceduration{#1}}

% ---------- Project Item ----------
\newcommand{\projectItem}[1]{
    \setkeys{project}{#1}
    \noindent
    \textbf{\projecttitle} \hfill \projectduration \\
    \ifthenelse{\equal{\projectkeyhighlight}{}}{}{%
        \textit{\projectkeyhighlight}
    }
    \vspace{2pt}
}

\define@key{project}{title}{\def\projecttitle{#1}}
\define@key{project}{duration}{\def\projectduration{#1}}
\define@key{project}{keyHighlight}{\def\projectkeyhighlight{#1}}

% Set defaults
\setkeys{project}{keyHighlight={}, duration={}}

\endinput
