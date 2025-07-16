import './globals.css'

export const metadata = {
  title: 'Resume Analyst',
  description: 'AI-Powered Resume Analysis Tool',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gray-100">{children}</body>
    </html>
  )
}
