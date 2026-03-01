"""
Knowledge Base for Career Readiness Recommendations
Contains learning resources, career advice, and project suggestions per skill.
"""

SKILL_KNOWLEDGE_BASE = {
    "artificial intelligence": {
        "description": "AI involves the simulation of human intelligence in machines, including learning, reasoning, and self-correction.",
        "learning_resources": [
            "Complete Andrew Ng's Machine Learning course on Coursera",
            "Study Deep Learning Specialization by deeplearning.ai",
            "Read 'Artificial Intelligence: A Modern Approach' by Russell & Norvig",
            "Practice with Kaggle competitions and datasets"
        ],
        "project_ideas": [
            "Build a chatbot using natural language processing",
            "Create an image classification system using CNNs",
            "Develop a recommendation system for e-commerce",
            "Implement predictive analytics for business metrics"
        ],
        "career_paths": ["AI Engineer", "ML Engineer", "Data Scientist", "AI Product Manager"],
        "estimated_time": "6-12 months for foundational proficiency"
    },
    "machine learning": {
        "description": "ML is a subset of AI focused on algorithms that learn from data without explicit programming.",
        "learning_resources": [
            "Complete Hands-On Machine Learning with Scikit-Learn and TensorFlow",
            "Take Google's Machine Learning Crash Course",
            "Study statistical foundations and linear algebra",
            "Practice on Kaggle and participate in competitions"
        ],
        "project_ideas": [
            "Build a fraud detection system",
            "Create a customer churn prediction model",
            "Develop a sentiment analysis tool",
            "Implement a time series forecasting model"
        ],
        "career_paths": ["ML Engineer", "Data Scientist", "AI Researcher"],
        "estimated_time": "4-8 months for intermediate skills"
    },
    "data analytics": {
        "description": "Data analytics involves examining datasets to draw conclusions and support decision-making.",
        "learning_resources": [
            "Master SQL for data querying and manipulation",
            "Learn Python libraries: pandas, numpy, matplotlib, seaborn",
            "Study Tableau or Power BI for visualization",
            "Take Google Data Analytics Professional Certificate"
        ],
        "project_ideas": [
            "Analyze sales data to identify trends and opportunities",
            "Create interactive dashboards for business metrics",
            "Perform cohort analysis for user retention",
            "Build automated reporting systems"
        ],
        "career_paths": ["Data Analyst", "Business Analyst", "Analytics Manager"],
        "estimated_time": "3-6 months for job-ready skills"
    },
    "product management": {
        "description": "Product management involves defining product vision, strategy, and coordinating cross-functional teams.",
        "learning_resources": [
            "Read 'Inspired' by Marty Cagan and 'The Lean Product Playbook'",
            "Take Product Management courses on Coursera or Reforge",
            "Learn product analytics tools (Mixpanel, Amplitude)",
            "Study agile methodologies and sprint planning"
        ],
        "project_ideas": [
            "Launch a side project from ideation to MVP",
            "Conduct user research and create product roadmaps",
            "Define and track product KPIs",
            "Create product requirement documents for real products"
        ],
        "career_paths": ["Product Manager", "Senior PM", "VP of Product", "Chief Product Officer"],
        "estimated_time": "6-12 months with hands-on experience"
    },
    "agile frameworks": {
        "description": "Agile is an iterative approach to project management emphasizing flexibility and collaboration.",
        "learning_resources": [
            "Get Certified Scrum Master (CSM) certification",
            "Study the Agile Manifesto and 12 principles",
            "Learn Scrum, Kanban, and SAFe methodologies",
            "Read 'Scrum: The Art of Doing Twice the Work in Half the Time'"
        ],
        "project_ideas": [
            "Facilitate sprint planning and retrospectives",
            "Implement Kanban boards for team workflow",
            "Lead daily standups and remove blockers",
            "Track velocity and optimize team performance"
        ],
        "career_paths": ["Scrum Master", "Agile Coach", "Product Owner", "Program Manager"],
        "estimated_time": "2-4 months for certification and practice"
    },
    "cross-functional team leadership": {
        "description": "Leading diverse teams across engineering, design, marketing, and other functions.",
        "learning_resources": [
            "Read 'The Five Dysfunctions of a Team' by Patrick Lencioni",
            "Study conflict resolution and stakeholder management",
            "Take leadership and communication courses",
            "Learn about OKRs and goal-setting frameworks"
        ],
        "project_ideas": [
            "Lead a cross-departmental initiative",
            "Facilitate alignment meetings between teams",
            "Create communication frameworks for distributed teams",
            "Mentor junior team members"
        ],
        "career_paths": ["Team Lead", "Engineering Manager", "Director", "VP"],
        "estimated_time": "Develops over 2-5 years with practice"
    },
    "kpi": {
        "description": "Key Performance Indicators are measurable values that demonstrate organizational effectiveness.",
        "learning_resources": [
            "Learn about North Star Metrics and AARRR framework",
            "Study data visualization and dashboard creation",
            "Master Excel/Google Sheets for metric tracking",
            "Read 'Lean Analytics' by Alistair Croll"
        ],
        "project_ideas": [
            "Define KPIs for a product or business",
            "Build automated KPI tracking dashboards",
            "Conduct A/B tests and measure impact on KPIs",
            "Create executive reports with key metrics"
        ],
        "career_paths": ["Product Analyst", "Business Analyst", "Product Manager"],
        "estimated_time": "2-3 months for foundational understanding"
    },
    "supply chain": {
        "description": "Supply chain management involves coordinating production, inventory, and distribution.",
        "learning_resources": [
            "Study supply chain fundamentals and logistics",
            "Learn demand forecasting and inventory optimization",
            "Master tools like SAP, Oracle SCM, or specialized software",
            "Take APICS certification courses"
        ],
        "project_ideas": [
            "Optimize inventory levels using forecasting models",
            "Implement just-in-time delivery systems",
            "Analyze supplier performance and costs",
            "Design distribution network optimization"
        ],
        "career_paths": ["Supply Chain Analyst", "Operations Manager", "Logistics Manager"],
        "estimated_time": "6-9 months for specialized knowledge"
    },
    "problem-solving": {
        "description": "Analytical and creative problem-solving skills for tackling complex challenges.",
        "learning_resources": [
            "Study frameworks: First Principles, 5 Whys, Root Cause Analysis",
            "Practice case interviews and structured thinking",
            "Read 'Thinking, Fast and Slow' by Daniel Kahneman",
            "Learn design thinking and creative problem-solving"
        ],
        "project_ideas": [
            "Solve real business problems using structured frameworks",
            "Participate in case competitions",
            "Debug complex technical issues systematically",
            "Facilitate problem-solving workshops"
        ],
        "career_paths": ["Consultant", "Product Manager", "Engineer", "Analyst"],
        "estimated_time": "Continuous skill development over career"
    },
    # ── Frontend Skills ──
    "react": {
        "description": "React is a JavaScript library for building fast, interactive user interfaces using a component-based architecture.",
        "learning_resources": [
            "Complete the official React documentation tutorial (react.dev)",
            "Build projects with React hooks, context API, and React Router",
            "Study state management with Redux Toolkit or Zustand",
            "Learn testing with React Testing Library and Jest"
        ],
        "project_ideas": [
            "Build a real-time dashboard with dynamic charts using React + D3.js",
            "Create a full-stack e-commerce app with cart, auth, and payment integration",
            "Develop a project management tool (Trello clone) with drag-and-drop",
            "Build a social media feed with infinite scroll and real-time updates"
        ],
        "career_paths": ["Frontend Developer", "React Developer", "Full-Stack Engineer", "UI Engineer"],
        "estimated_time": "3-5 months for production-ready proficiency"
    },
    "angular": {
        "description": "Angular is a TypeScript-based frontend framework by Google for building scalable single-page applications.",
        "learning_resources": [
            "Complete the official Angular tutorial (angular.io)",
            "Master RxJS observables, pipes, and reactive forms",
            "Study Angular modules, lazy loading, and route guards",
            "Learn NgRx for state management and Angular Material for UI"
        ],
        "project_ideas": [
            "Build an enterprise admin dashboard with role-based access control",
            "Create a real-time chat application using Angular + WebSockets",
            "Develop a CRM system with complex forms, validation, and data grids",
            "Build a movie/show discovery app with search, filters, and pagination"
        ],
        "career_paths": ["Angular Developer", "Frontend Engineer", "Full-Stack Developer", "Enterprise UI Developer"],
        "estimated_time": "4-6 months for intermediate proficiency"
    },
    "javascript": {
        "description": "JavaScript is the core programming language of the web, essential for frontend interactivity and increasingly used in backend development.",
        "learning_resources": [
            "Master ES6+ features: arrow functions, destructuring, async/await, modules",
            "Study 'You Don't Know JS' book series by Kyle Simpson",
            "Learn DOM manipulation, event handling, and browser APIs",
            "Practice algorithmic problem-solving on LeetCode/HackerRank with JS"
        ],
        "project_ideas": [
            "Build a vanilla JS single-page application without any framework",
            "Create a browser-based game (e.g., Snake, 2048) using Canvas API",
            "Develop a Chrome extension with content scripts and popup UI",
            "Build a real-time collaborative text editor using WebSockets"
        ],
        "career_paths": ["Frontend Developer", "Full-Stack Developer", "JavaScript Engineer"],
        "estimated_time": "2-4 months for solid intermediate skills"
    },
    "typescript": {
        "description": "TypeScript adds static typing to JavaScript, enabling better tooling, refactoring, and large-scale application development.",
        "learning_resources": [
            "Complete the TypeScript Handbook on typescriptlang.org",
            "Study advanced types: generics, mapped types, conditional types, utility types",
            "Practice converting JavaScript projects to TypeScript",
            "Learn TypeScript with React (typed hooks, props interfaces) or Angular"
        ],
        "project_ideas": [
            "Migrate an existing JS project to TypeScript with strict mode",
            "Build a type-safe REST API client with generics and Zod validation",
            "Create a design system with fully typed component props",
            "Develop a CLI tool in TypeScript with commander.js"
        ],
        "career_paths": ["Frontend Engineer", "Full-Stack Developer", "TypeScript Developer"],
        "estimated_time": "2-3 months if JavaScript proficient"
    },
    "css": {
        "description": "CSS (Cascading Style Sheets) controls the visual presentation of web pages including layout, colors, typography, and animations.",
        "learning_resources": [
            "Master Flexbox and CSS Grid layout systems thoroughly",
            "Study responsive design with media queries and mobile-first approach",
            "Learn CSS animations, transitions, and keyframes",
            "Practice with CSS frameworks: Tailwind CSS, Bootstrap, or CSS Modules"
        ],
        "project_ideas": [
            "Build a fully responsive portfolio website with advanced animations",
            "Recreate popular website layouts (Airbnb, Spotify) pixel-perfect",
            "Create a CSS-only component library with buttons, cards, modals",
            "Build a responsive email template system"
        ],
        "career_paths": ["Frontend Developer", "UI Developer", "Web Designer", "UX Engineer"],
        "estimated_time": "2-3 months for advanced proficiency"
    },
    "html": {
        "description": "HTML is the standard markup language for creating web pages, providing structure and semantic meaning to web content.",
        "learning_resources": [
            "Master semantic HTML5 elements (article, section, nav, aside)",
            "Study web accessibility (WCAG guidelines, ARIA attributes)",
            "Learn SEO best practices and structured data markup",
            "Practice with HTML forms, validation, and multimedia elements"
        ],
        "project_ideas": [
            "Build an accessible multi-page website following WCAG 2.1 AA standards",
            "Create a documentation site with proper semantic structure",
            "Develop an HTML email template system",
            "Build interactive forms with client-side validation"
        ],
        "career_paths": ["Frontend Developer", "Web Developer", "Accessibility Specialist"],
        "estimated_time": "1-2 months for advanced proficiency"
    },
    "vue": {
        "description": "Vue.js is a progressive JavaScript framework for building user interfaces, known for its gentle learning curve and flexibility.",
        "learning_resources": [
            "Complete the official Vue 3 tutorial with Composition API",
            "Study Pinia for state management and Vue Router",
            "Learn Nuxt.js for server-side rendering and static generation",
            "Practice with Vuetify or PrimeVue component libraries"
        ],
        "project_ideas": [
            "Build a recipe sharing app with CRUD, search, and user profiles",
            "Create a kanban board with drag-and-drop using Vue + Pinia",
            "Develop a blog platform with Nuxt.js and markdown support",
            "Build a weather dashboard with real-time API data and charts"
        ],
        "career_paths": ["Vue Developer", "Frontend Engineer", "Full-Stack Developer"],
        "estimated_time": "3-4 months for production-ready skills"
    },
    # ── Backend Skills ──
    "node.js": {
        "description": "Node.js is a JavaScript runtime for building scalable server-side applications with non-blocking I/O.",
        "learning_resources": [
            "Master Express.js or Fastify for REST API development",
            "Study async patterns: callbacks, promises, async/await, event loop",
            "Learn database integration with PostgreSQL (Prisma/Sequelize) and MongoDB",
            "Practice building microservices with message queues (RabbitMQ/Redis)"
        ],
        "project_ideas": [
            "Build a RESTful API with authentication, rate limiting, and caching",
            "Create a real-time notification system using Socket.io",
            "Develop a file upload service with S3 integration and image processing",
            "Build a URL shortener with analytics tracking"
        ],
        "career_paths": ["Backend Developer", "Node.js Developer", "Full-Stack Engineer", "API Developer"],
        "estimated_time": "3-5 months for production-ready skills"
    },
    "python": {
        "description": "Python is a versatile programming language used in web development, data science, automation, and AI/ML.",
        "learning_resources": [
            "Master Python fundamentals: data structures, OOP, decorators, generators",
            "Learn Django or FastAPI for web application development",
            "Study testing with pytest and type hints with mypy",
            "Practice data manipulation with pandas, numpy"
        ],
        "project_ideas": [
            "Build a REST API with FastAPI including JWT auth and database models",
            "Create a web scraper with data pipeline and automated reporting",
            "Develop a task automation tool for workflow optimization",
            "Build a data analytics dashboard with Streamlit"
        ],
        "career_paths": ["Python Developer", "Backend Engineer", "Data Engineer", "ML Engineer"],
        "estimated_time": "3-5 months for intermediate proficiency"
    },
    "sql": {
        "description": "SQL is the standard language for managing and querying relational databases.",
        "learning_resources": [
            "Master JOINs, subqueries, CTEs, window functions",
            "Study query optimization, indexing strategies, and execution plans",
            "Learn database design: normalization, ERD diagrams, constraints",
            "Practice on LeetCode SQL problems and real-world datasets"
        ],
        "project_ideas": [
            "Design and implement a normalized database schema for an e-commerce system",
            "Build complex analytical queries for business intelligence reporting",
            "Create stored procedures and triggers for data integrity",
            "Optimize slow queries in an existing database"
        ],
        "career_paths": ["Database Developer", "Data Analyst", "Backend Developer", "Data Engineer"],
        "estimated_time": "2-4 months for advanced proficiency"
    },
    "docker": {
        "description": "Docker enables containerization of applications for consistent deployment across environments.",
        "learning_resources": [
            "Master Dockerfile creation, multi-stage builds, and image optimization",
            "Study Docker Compose for multi-container application orchestration",
            "Learn container networking, volumes, and security best practices",
            "Practice CI/CD pipeline integration with Docker builds"
        ],
        "project_ideas": [
            "Containerize a full-stack application (frontend + backend + database)",
            "Build a microservices architecture with Docker Compose",
            "Create a CI/CD pipeline that builds, tests, and deploys Docker images",
            "Set up a local development environment with hot reload in containers"
        ],
        "career_paths": ["DevOps Engineer", "Platform Engineer", "Backend Developer", "SRE"],
        "estimated_time": "2-3 months for production usage"
    },
    "kubernetes": {
        "description": "Kubernetes is an open-source container orchestration platform for automating deployment, scaling, and management.",
        "learning_resources": [
            "Study Kubernetes architecture: pods, services, deployments, ingress",
            "Complete CKA (Certified Kubernetes Administrator) preparation",
            "Learn Helm charts for package management",
            "Practice with Minikube or Kind for local cluster development"
        ],
        "project_ideas": [
            "Deploy a microservices application on Kubernetes with auto-scaling",
            "Build a CI/CD pipeline with ArgoCD for GitOps deployments",
            "Create Helm charts for your application stack",
            "Implement blue-green and canary deployment strategies"
        ],
        "career_paths": ["DevOps Engineer", "Platform Engineer", "SRE", "Cloud Architect"],
        "estimated_time": "4-6 months for intermediate proficiency"
    },
    "aws": {
        "description": "Amazon Web Services is the leading cloud platform offering compute, storage, database, and AI/ML services.",
        "learning_resources": [
            "Study for AWS Solutions Architect Associate certification",
            "Master core services: EC2, S3, RDS, Lambda, API Gateway, CloudFront",
            "Learn Infrastructure as Code with CloudFormation or Terraform",
            "Practice with AWS Free Tier hands-on labs"
        ],
        "project_ideas": [
            "Build a serverless API with Lambda, API Gateway, and DynamoDB",
            "Create a static website with S3, CloudFront, and Route 53",
            "Deploy a scalable web application with ECS/EKS and RDS",
            "Build an event-driven architecture with SQS, SNS, and Lambda"
        ],
        "career_paths": ["Cloud Engineer", "Solutions Architect", "DevOps Engineer", "Cloud Developer"],
        "estimated_time": "4-6 months for associate-level proficiency"
    },
    "microsoft azure": {
        "description": "Microsoft Azure is a comprehensive cloud platform for building, deploying, and managing applications and services.",
        "learning_resources": [
            "Study for Azure Fundamentals (AZ-900) certification",
            "Master Azure App Service, Azure Functions, Azure SQL, Blob Storage",
            "Learn Azure DevOps pipelines and ARM/Bicep templates",
            "Practice with Azure free tier and Microsoft Learn modules"
        ],
        "project_ideas": [
            "Build a serverless application with Azure Functions and Cosmos DB",
            "Create a CI/CD pipeline using Azure DevOps with staging environments",
            "Deploy a containerized app with Azure Kubernetes Service (AKS)",
            "Build a data pipeline with Azure Data Factory and Synapse Analytics"
        ],
        "career_paths": ["Azure Cloud Engineer", "Solutions Architect", "DevOps Engineer", "Cloud Developer"],
        "estimated_time": "4-6 months for fundamentals + associate certification"
    },
    "git": {
        "description": "Git is the industry-standard distributed version control system for tracking code changes and team collaboration.",
        "learning_resources": [
            "Master branching strategies: GitFlow, trunk-based development",
            "Study rebasing, cherry-picking, bisect, and stash workflows",
            "Learn GitHub/GitLab features: PRs, code reviews, CI integration",
            "Practice resolving merge conflicts in real projects"
        ],
        "project_ideas": [
            "Contribute to an open-source project following proper Git workflow",
            "Set up branch protection rules and automated PR checks",
            "Create a Git hooks system for code quality enforcement",
            "Manage a monorepo with multiple packages using Git subtrees"
        ],
        "career_paths": ["Software Developer", "DevOps Engineer", "Technical Lead"],
        "estimated_time": "2-4 weeks for advanced proficiency"
    },
    "rest api": {
        "description": "REST API design involves creating scalable, stateless web services following HTTP standards and resource-oriented architecture.",
        "learning_resources": [
            "Study RESTful design principles: resources, HTTP methods, status codes",
            "Learn API documentation with OpenAPI/Swagger",
            "Master authentication: OAuth 2.0, JWT, API keys",
            "Study API versioning, pagination, filtering, and error handling patterns"
        ],
        "project_ideas": [
            "Design and build a REST API for a task management application",
            "Create an API gateway with rate limiting and authentication",
            "Build a public API with documentation, SDKs, and developer portal",
            "Implement HATEOAS-driven API with hypermedia links"
        ],
        "career_paths": ["Backend Developer", "API Developer", "Full-Stack Engineer", "Solutions Architect"],
        "estimated_time": "2-3 months for production-quality API design"
    },
    "graphql": {
        "description": "GraphQL is a query language for APIs that lets clients request exactly the data they need, reducing over-fetching.",
        "learning_resources": [
            "Master GraphQL schema design: types, queries, mutations, subscriptions",
            "Learn Apollo Client and Apollo Server or Yoga",
            "Study DataLoader for batching and caching N+1 query problems",
            "Practice with GraphQL Playground and code-first schema generation"
        ],
        "project_ideas": [
            "Build a GraphQL API for a social media platform with subscriptions",
            "Create a federated GraphQL gateway for microservices",
            "Develop a full-stack app with Apollo Client + React + GraphQL",
            "Migrate an existing REST API to GraphQL"
        ],
        "career_paths": ["Backend Developer", "Full-Stack Engineer", "API Architect"],
        "estimated_time": "2-4 months for production usage"
    },
    # ── Data & Analytics ──
    "power bi": {
        "description": "Power BI is Microsoft's business analytics tool for creating interactive dashboards and data visualizations.",
        "learning_resources": [
            "Complete Microsoft Power BI Data Analyst (PL-300) certification path",
            "Master DAX formulas for calculated columns and measures",
            "Study Power Query (M language) for data transformation",
            "Learn data modeling: star schema, relationships, row-level security"
        ],
        "project_ideas": [
            "Build an executive sales dashboard with drill-through and bookmarks",
            "Create a customer analytics report with cohort analysis and RFM segmentation",
            "Develop a financial reporting dashboard with YoY comparison and forecasting",
            "Build a real-time operational dashboard connected to streaming data"
        ],
        "career_paths": ["BI Developer", "Data Analyst", "Business Analyst", "Analytics Manager"],
        "estimated_time": "3-4 months for PL-300 certification readiness"
    },
    "tableau": {
        "description": "Tableau is a leading data visualization platform for creating interactive, shareable dashboards.",
        "learning_resources": [
            "Study for Tableau Desktop Specialist or Certified Data Analyst",
            "Master calculated fields, LOD expressions, and table calculations",
            "Learn Tableau Prep for data preparation and blending",
            "Practice with real datasets from Kaggle or data.gov"
        ],
        "project_ideas": [
            "Build an interactive COVID/health data dashboard with map visualizations",
            "Create a marketing analytics dashboard with campaign ROI tracking",
            "Develop a supply chain visibility dashboard with KPI scorecards",
            "Build a storytelling dashboard for executive presentations"
        ],
        "career_paths": ["Data Analyst", "BI Developer", "Data Visualization Specialist"],
        "estimated_time": "2-4 months for certification-ready skills"
    },
    # ── Product & Project Management ──
    "jira": {
        "description": "Jira is Atlassian's project management tool for agile teams, used for sprint planning, backlog management, and issue tracking.",
        "learning_resources": [
            "Master Jira board configuration: Scrum boards, Kanban boards",
            "Study JQL (Jira Query Language) for advanced filtering",
            "Learn workflow customization, automation rules, and integrations",
            "Practice sprint planning, velocity tracking, and burndown charts"
        ],
        "project_ideas": [
            "Set up a complete Jira project with epics, stories, and sprint cycles",
            "Create custom Jira dashboards for team velocity and sprint health",
            "Build Jira automation rules for status transitions and notifications",
            "Design a cross-team project tracking system with Jira portfolios"
        ],
        "career_paths": ["Project Manager", "Scrum Master", "Product Manager", "Jira Administrator"],
        "estimated_time": "2-4 weeks for effective daily usage"
    },
    "product lifecycle management": {
        "description": "Product lifecycle management covers the entire journey of a product from ideation through development, launch, growth, and sunset.",
        "learning_resources": [
            "Study product strategy frameworks: Jobs-to-be-Done, Blue Ocean Strategy",
            "Learn roadmap planning tools: Productboard, Aha!, ProductPlan",
            "Master product discovery techniques: design sprints, user story mapping",
            "Read 'Inspired' by Marty Cagan and 'Continuous Discovery Habits' by Teresa Torres"
        ],
        "project_ideas": [
            "Create a complete product strategy document for a SaaS application",
            "Build a product roadmap with prioritization using RICE/ICE scoring",
            "Conduct user research and synthesize findings into actionable product requirements",
            "Design a go-to-market strategy for a new product feature"
        ],
        "career_paths": ["Product Manager", "Senior PM", "Director of Product", "Chief Product Officer"],
        "estimated_time": "6-9 months with hands-on product ownership"
    },
    "a/b testing": {
        "description": "A/B testing is a controlled experiment methodology to compare two versions and determine which performs better based on statistical significance.",
        "learning_resources": [
            "Study experimental design: hypothesis formation, sample size calculation, statistical significance",
            "Learn tools: Optimizely, Google Optimize, LaunchDarkly, or Statsig",
            "Master statistical concepts: p-values, confidence intervals, power analysis, Bayesian vs frequentist",
            "Read 'Trustworthy Online Controlled Experiments' by Kohavi, Tang & Xu"
        ],
        "project_ideas": [
            "Design and run an A/B test on a landing page with conversion tracking",
            "Build an A/B testing framework with automated statistical analysis",
            "Create a feature flagging system with gradual rollout capabilities",
            "Analyze historical A/B test results and present findings with effect sizes"
        ],
        "career_paths": ["Growth Product Manager", "Data Scientist", "Experimentation Analyst", "Growth Engineer"],
        "estimated_time": "2-3 months for practical experimentation skills"
    },
    "product roadmap": {
        "description": "Product roadmapping is the strategic process of defining product vision, priorities, and delivery timeline for stakeholder alignment.",
        "learning_resources": [
            "Study roadmap frameworks: Now-Next-Later, outcome-based roadmaps",
            "Learn prioritization techniques: RICE, ICE, MoSCoW, Kano model",
            "Master roadmap tools: Productboard, Aha!, Linear, Notion",
            "Read 'Product Roadmaps Relaunched' by Lombardo et al."
        ],
        "project_ideas": [
            "Create a 6-month product roadmap with OKR alignment for a SaaS product",
            "Build a prioritization scoring system for feature requests",
            "Design a stakeholder communication cadence with roadmap review presentations",
            "Develop a data-driven roadmap using customer feedback analysis"
        ],
        "career_paths": ["Product Manager", "Technical Product Manager", "Director of Product"],
        "estimated_time": "2-4 months of hands-on practice"
    },
    # ── DevOps & CI/CD ──
    "ci/cd": {
        "description": "CI/CD (Continuous Integration/Continuous Deployment) automates the build, test, and deployment pipeline for faster, reliable software delivery.",
        "learning_resources": [
            "Master GitHub Actions, GitLab CI, or Jenkins pipeline configuration",
            "Study deployment strategies: blue-green, canary, rolling updates",
            "Learn infrastructure as code: Terraform, Ansible, or Pulumi",
            "Practice automated testing integration: unit, integration, E2E"
        ],
        "project_ideas": [
            "Build a complete CI/CD pipeline with automated testing and staging deployment",
            "Create a multi-environment deployment system (dev/staging/prod)",
            "Implement automated security scanning (SAST/DAST) in the pipeline",
            "Build a self-service deployment platform for development teams"
        ],
        "career_paths": ["DevOps Engineer", "SRE", "Platform Engineer", "Release Manager"],
        "estimated_time": "3-4 months for production pipeline management"
    },
    # ── Testing ──
    "testing": {
        "description": "Software testing encompasses strategies and tools for ensuring code quality through unit, integration, E2E, and performance testing.",
        "learning_resources": [
            "Master testing pyramid: unit tests, integration tests, E2E tests",
            "Learn testing frameworks: Jest, Pytest, JUnit, Cypress, Playwright",
            "Study TDD (Test-Driven Development) and BDD methodologies",
            "Practice mocking, stubbing, and test data management"
        ],
        "project_ideas": [
            "Add comprehensive test coverage to an existing project (target 80%+)",
            "Build an automated E2E test suite for a web application with Cypress/Playwright",
            "Create a performance testing setup with k6 or JMeter",
            "Implement a visual regression testing pipeline with Percy or Chromatic"
        ],
        "career_paths": ["QA Engineer", "SDET", "Test Automation Engineer", "Quality Lead"],
        "estimated_time": "2-4 months for comprehensive testing skills"
    },
    # ── Soft/Domain Skills ──
    "demand forecasting": {
        "description": "Demand forecasting uses statistical methods and ML to predict future customer demand for inventory and resource planning.",
        "learning_resources": [
            "Study time series models: ARIMA, SARIMA, Prophet, exponential smoothing",
            "Learn ML approaches: XGBoost, LSTM for sequential forecasting",
            "Master evaluation metrics: MAPE, RMSE, MAE, forecast bias",
            "Practice with real retail/supply chain datasets on Kaggle"
        ],
        "project_ideas": [
            "Build a demand forecasting model for retail inventory optimization",
            "Create a multi-product forecasting dashboard with confidence intervals",
            "Develop an automated demand planning system with anomaly detection",
            "Compare classical time series vs. ML models on real business data"
        ],
        "career_paths": ["Demand Planner", "Supply Chain Analyst", "Data Scientist", "Operations Research Analyst"],
        "estimated_time": "3-5 months for production-level forecasting"
    },
    "product backlog": {
        "description": "Product backlog management involves maintaining a prioritized list of features, bugs, and technical debt items for agile delivery.",
        "learning_resources": [
            "Study backlog refinement techniques and user story writing (INVEST criteria)",
            "Learn estimation methods: story points, T-shirt sizing, planning poker",
            "Master tools: Jira, Linear, Azure DevOps for backlog management",
            "Read 'User Story Mapping' by Jeff Patton"
        ],
        "project_ideas": [
            "Create and manage a product backlog for a mobile app from scratch",
            "Build an automated backlog health dashboard tracking age, priorities, and velocity",
            "Design acceptance criteria and definition of done templates",
            "Run a backlog grooming workshop and document the process"
        ],
        "career_paths": ["Product Owner", "Product Manager", "Scrum Master", "Business Analyst"],
        "estimated_time": "1-2 months for effective backlog management"
    },
    "default": {
        "description": "This skill is valuable for professional growth and career advancement.",
        "learning_resources": [
            "Search for online courses on Coursera, Udemy, or LinkedIn Learning",
            "Read industry-specific books and blogs",
            "Join professional communities and attend conferences",
            "Find mentors in your target field"
        ],
        "project_ideas": [
            "Build a portfolio project demonstrating this skill",
            "Contribute to open-source projects",
            "Create case studies from your work experience",
            "Share knowledge through blog posts or presentations"
        ],
        "career_paths": ["Specialist roles in this domain"],
        "estimated_time": "Varies by skill complexity (3-12 months)"
    }
}


def get_skill_knowledge(skill):
    """
    Retrieve knowledge base entry for a skill.
    Returns default entry if skill not found.
    """
    skill_lower = skill.lower().strip()
    
    # Try exact match first
    if skill_lower in SKILL_KNOWLEDGE_BASE:
        return SKILL_KNOWLEDGE_BASE[skill_lower]
    
    # Try partial match (e.g., "monitor kpis" matches "kpi")
    for key in SKILL_KNOWLEDGE_BASE.keys():
        if key in skill_lower or skill_lower in key:
            return SKILL_KNOWLEDGE_BASE[key]
    
    # Return default
    return SKILL_KNOWLEDGE_BASE["default"]


def get_all_knowledge_texts():
    """Get all knowledge base content as text documents for embedding."""
    documents = []
    for skill, knowledge in SKILL_KNOWLEDGE_BASE.items():
        if skill == "default":
            continue
        
        doc_text = f"Skill: {skill}\n"
        doc_text += f"Description: {knowledge['description']}\n"
        doc_text += f"Learning Resources: {', '.join(knowledge['learning_resources'])}\n"
        doc_text += f"Project Ideas: {', '.join(knowledge['project_ideas'])}\n"
        doc_text += f"Career Paths: {', '.join(knowledge['career_paths'])}\n"
        
        documents.append({
            "skill": skill,
            "text": doc_text,
            "metadata": knowledge
        })
    
    return documents
