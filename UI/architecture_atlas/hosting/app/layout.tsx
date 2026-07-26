export const metadata = {
  title: "Cynolycus // System Atlas",
  description: "Interactive architecture atlas for the CynolycusBot system.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body style={{ margin: 0 }}>{children}</body>
    </html>
  );
}
